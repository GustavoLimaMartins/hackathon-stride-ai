"""Motor de integração com o LLM para mapeamento STRIDE e prescrição de contramedidas.

Dois papéis de modelo: 'rewriter' (gpt-4o) interpreta/reescreve os dados brutos
do JSON estruturado antes da análise, adicionando um resumo narrativo de
contexto; 'analyst' (gpt-5) elabora o parecer STRIDE final com as contramedidas.

Cada papel é abstraído por uma interface própria, com duas implementações de
provedor (interface uniforme para o main.py):
  - Analyst -> StrideReport estruturado: OpenAIAnalyst (gpt-5, LangChain
    with_structured_output) e OllamaAnalyst (gemma3:12b, format=schema).
  - Rewriter -> grafo + 'narrative_summary': OpenAIRewriter (gpt-4o) e
    OllamaRewriter (gemma3:12b).
Todas as chamadas emitem um registro de auditoria (src/llm_audit).

load_primary_analyst()/load_primary_rewriter() decidem o provedor principal de
cada papel conforme a env var LLM_MODEL_PAID (default true = OpenAI/pago).
Quando o principal é a OpenAI, o main.py ainda usa o Ollama como FALLBACK no
timeout de cada etapa, independente da flag.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src import llm_audit
from src.prompts import build_rewriter_prompt
from src.stride_models import StrideReport

_MODELS = {
    "rewriter": "gpt-4o",
    "analyst": "gpt-5",
}

# Esforço de raciocínio do 'analyst' (gpt-5, modelo de reasoning). Fixado
# explicitamente em 'medium': deixar o parâmetro implícito expõe o pipeline a
# mudanças de default da API. 'medium' equilibra profundidade e latência,
# preservando as justificativas/contramedidas específicas exigidas pelo prompt.
_ANALYST_REASONING_EFFORT = "medium"

# Timeout (segundos) da chamada ao 'analyst'. Diagnóstico (2026-07-20): o tempo
# de resposta do gpt-5 é dominado pela latência da OpenAI no momento/carga —
# grafos densos que antes voltavam em ~10 min passaram a estourar isso sem
# nenhuma mudança no código (a troca para gpt-5-mini não melhorou, confirmando
# a origem externa). Sem timeout explícito o app trava indefinidamente numa
# janela de latência ruim. Com este teto de 10 min, a chamada falha de forma
# limpa (APITimeoutError) e a UI mostra uma mensagem clara em vez de congelar.
# max_retries=0 evita que o SDK reenvie a chamada após o timeout, o que
# multiplicaria a espera (10 min × tentativas).
# Configurável via .env (ANALYST_TIMEOUT_SECONDS) para ajustar sem editar
# código — ex.: forçar um valor baixo para testar o fallback Ollama.
_ANALYST_TIMEOUT_SECONDS = int(os.environ.get("ANALYST_TIMEOUT_SECONDS", "600"))
_ANALYST_MAX_RETRIES = 0

# Timeout (segundos) da chamada ao 'rewriter' (gpt-4o). É uma síntese curta
# texto->texto (não reasoning), então um teto bem menor que o do analyst basta;
# no estouro, a etapa degrada graciosamente (segue sem o resumo). Configurável
# via .env (REWRITER_TIMEOUT_SECONDS).
_REWRITER_TIMEOUT_SECONDS = int(os.environ.get("REWRITER_TIMEOUT_SECONDS", "60"))

# --- Escolha de provedor: pago (OpenAI) vs. local/open-source (Ollama) -------
# LLM_MODEL_PAID controla qual provedor é o principal:
#   true  (default) -> OpenAI (gpt-5) é o principal; no timeout, cai para o
#                       Ollama como FALLBACK (comportamento de contingência).
#   false            -> Ollama (gemma3:12b) é usado direto como principal, sem
#                       tentar a OpenAI nem exigir OPENAI_API_KEY. Útil para
#                       rodar 100% offline/sem custo de API.
_LLM_MODEL_PAID = os.environ.get("LLM_MODEL_PAID", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)

# --- Fallback local (Ollama) -------------------------------------------------
# Modelo local usado quando o 'analyst' da OpenAI dá timeout (ou como principal
# se LLM_MODEL_PAID=false). gemma3:12b (~8 GB) roda em CPU sem GPU dedicada e
# cabe em máquinas modestas; baixe com `ollama pull gemma3:12b`. Em hardware com
# mais folga (disco/RAM/GPU) troque por `gemma3:27b` para um parecer mais rico.
# O host é configurável por env para apontar a um daemon Ollama remoto se
# necessário.
_OLLAMA_ANALYST_MODEL = "gemma3:12b"
_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# Janela de contexto do Ollama: o system prompt STRIDE (~10k chars) + o grafo
# JSON (~5k+) precisam caber com folga. gemma3 suporta 128K; 16384 tokens já
# acomodam prompt + parecer estruturado sem estourar a memória à toa.
_OLLAMA_NUM_CTX = 16384

_llm_cache: dict[str, ChatOpenAI] = {}


def load_llm(role: str) -> ChatOpenAI:
    """Carrega (e cacheia) o cliente LLM correspondente ao papel informado.

    role: 'rewriter' (gpt-4o, interpreta os dados do JSON) ou 'analyst'
    (gpt-5, elabora o parecer STRIDE final).
    """
    if role not in _MODELS:
        raise ValueError(
            f"papel de LLM desconhecido: {role!r} (esperado um de {sorted(_MODELS)})"
        )

    if role in _llm_cache:
        return _llm_cache[role]

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY não configurada. Defina essa variável de ambiente "
            "(veja .env.example) antes de usar o stride_engine."
        )

    kwargs: dict[str, Any] = {"model": _MODELS[role]}
    if role == "analyst":
        kwargs["reasoning_effort"] = _ANALYST_REASONING_EFFORT
        kwargs["timeout"] = _ANALYST_TIMEOUT_SECONDS
        kwargs["max_retries"] = _ANALYST_MAX_RETRIES
    elif role == "rewriter":
        # gpt-4o não é modelo de reasoning: sem reasoning_effort. Timeout curto
        # próprio; sem retries para não multiplicar a espera de uma etapa não
        # crítica (o app segue sem o resumo se ela falhar).
        kwargs["timeout"] = _REWRITER_TIMEOUT_SECONDS
        kwargs["max_retries"] = 0

    llm = ChatOpenAI(**kwargs)
    _llm_cache[role] = llm
    return llm


def load_analyst_structured() -> Any:
    """Cliente 'analyst' que devolve um StrideReport tipado, não texto livre.

    Envolve o LLM com with_structured_output(StrideReport): o schema Pydantic é
    injetado na chamada e a resposta já vem como um StrideReport validado (lista
    de riscos, cada um ancorado a um id do grafo). É o que sustenta a
    rastreabilidade visual risco -> bounding box na UI.

    Mantido por compatibilidade; o main.py usa a abstração BaseAnalyst abaixo.
    """
    return load_llm("analyst").with_structured_output(StrideReport)


# --- Abstração de analyst (OpenAI padrão + Ollama fallback) -------------------


class BaseAnalyst:
    """Analista STRIDE: recebe os prompts como strings e devolve um StrideReport.

    A interface é agnóstica de provedor (LangChain/OpenAI vs Ollama), então o
    main.py troca um pelo outro sem saber qual está por baixo. As subclasses
    envolvem a chamada ao LLM num contexto de auditoria (src/llm_audit).
    """

    provider: str = ""
    model: str = ""
    is_fallback: bool = False

    def analyze(self, system_prompt: str, user_message: str) -> StrideReport:
        raise NotImplementedError


class OpenAIAnalyst(BaseAnalyst):
    """Analista padrão: gpt-5 via LangChain, saída estruturada StrideReport."""

    provider = "openai"
    model = _MODELS["analyst"]
    is_fallback = False

    def analyze(self, system_prompt: str, user_message: str) -> StrideReport:
        # include_raw=True devolve {'parsed': StrideReport, 'raw': AIMessage} —
        # o 'raw' carrega usage_metadata (tokens), que a auditoria registra.
        analyst = load_llm("analyst").with_structured_output(
            StrideReport, include_raw=True
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        with llm_audit.audit_analysis(
            self.provider, self.model, self.is_fallback
        ) as audit:
            result = analyst.invoke(messages)
            report: StrideReport = result["parsed"]
            usage = getattr(result.get("raw"), "usage_metadata", None) or {}
            audit["input_tokens"] = usage.get("input_tokens")
            audit["output_tokens"] = usage.get("output_tokens")
            audit["n_risks"] = len(report.risks)
        return report


class OllamaAnalyst(BaseAnalyst):
    """Local: gemma3:12b via Ollama, com o schema StrideReport forçado.

    O Ollama aceita `format=<json schema>` e devolve o conteúdo já no formato do
    schema; parseamos com StrideReport.model_validate_json — mantendo a
    rastreabilidade risco->bbox (cada risco traz o id do elemento). Não é modelo
    de reasoning: controle via options (temperature=0 p/ determinismo, num_ctx
    p/ caber prompt+parecer). Roda offline.

    is_fallback distingue o papel na auditoria: True (default) quando usado como
    contingência do timeout da OpenAI; False quando é o analyst PRINCIPAL (ver
    LLM_MODEL_PAID=false em load_primary_analyst).
    """

    provider = "ollama"
    model = _OLLAMA_ANALYST_MODEL

    def __init__(self, is_fallback: bool = True):
        self.is_fallback = is_fallback

    def analyze(self, system_prompt: str, user_message: str) -> StrideReport:
        # Import tardio: o 'ollama' é dependência do fallback; não deve ser
        # exigido no caminho feliz (nem no CI) se o fallback nunca dispara.
        from ollama import Client

        client = Client(host=os.environ.get("OLLAMA_HOST", _OLLAMA_DEFAULT_HOST))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        with llm_audit.audit_analysis(
            self.provider, self.model, self.is_fallback
        ) as audit:
            # Ollama não expõe contagem de tokens do mesmo jeito; registramos o
            # tamanho do prompt (proxy) e o nº de riscos gerados.
            audit["prompt_chars"] = len(system_prompt) + len(user_message)
            response = client.chat(
                model=self.model,
                messages=messages,
                format=StrideReport.model_json_schema(),
                options={"temperature": 0, "num_ctx": _OLLAMA_NUM_CTX},
            )
            report = StrideReport.model_validate_json(response.message.content)
            audit["n_risks"] = len(report.risks)
        return report


def load_openai_analyst() -> OpenAIAnalyst:
    """Analista pago (gpt-5). Principal quando LLM_MODEL_PAID=true (default)."""
    return OpenAIAnalyst()


def load_ollama_analyst(is_fallback: bool = True) -> OllamaAnalyst:
    """Analista local (gemma3:12b via Ollama).

    is_fallback=True (default): usado como contingência no timeout da OpenAI.
    is_fallback=False: usado como principal (LLM_MODEL_PAID=false), sem tentar
    a OpenAI antes.
    """
    return OllamaAnalyst(is_fallback=is_fallback)


def load_primary_analyst() -> BaseAnalyst:
    """Analyst principal da análise STRIDE, conforme LLM_MODEL_PAID.

    true (default): OpenAI (gpt-5) — o timeout aciona o fallback Ollama em
    main.py. false: Ollama (gemma3:12b) direto como principal, sem tentar a
    OpenAI nem exigir OPENAI_API_KEY — para rodar 100% local/sem custo de API.
    """
    if _LLM_MODEL_PAID:
        return load_openai_analyst()
    return load_ollama_analyst(is_fallback=False)


# --- Abstração de rewriter (enriquece o grafo antes do analyst) ---------------
#
# Espelha a abstração de analyst, mas com contrato dict->dict: recebe o grafo de
# graph_builder.to_json() e devolve uma CÓPIA com a chave adicional
# 'narrative_summary' (resumo textual do padrão arquitetural). É uma etapa de
# CONTEXTO auxiliar — nunca remove/renomeia chaves nem altera os ids (cN/bN) de
# que o analyst e o group_risks dependem para a rastreabilidade risco->bbox.


class BaseRewriter:
    """Enriquece o grafo com um resumo narrativo antes do 'analyst'.

    Contrato dict->dict (não strings): recebe o grafo de to_json() e devolve
    uma cópia com a chave 'narrative_summary'. As subclasses montam o próprio
    prompt (build_rewriter_prompt) e envolvem a chamada num contexto de
    auditoria (src/llm_audit), como os analysts.
    """

    provider: str = ""
    model: str = ""
    is_fallback: bool = False

    def rewrite(self, graph: dict) -> dict:
        raise NotImplementedError


class OpenAIRewriter(BaseRewriter):
    """Rewriter pago: gpt-4o via LangChain, saída textual (resumo)."""

    provider = "openai"
    model = _MODELS["rewriter"]
    is_fallback = False

    def rewrite(self, graph: dict) -> dict:
        llm = load_llm("rewriter")
        system_prompt, user_message = build_rewriter_prompt(graph)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        with llm_audit.audit_analysis(
            self.provider, self.model, self.is_fallback
        ) as audit:
            result = llm.invoke(messages)
            narrative = (result.content or "").strip()
            audit["narrative_chars"] = len(narrative)
        return {**graph, "narrative_summary": narrative}


class OllamaRewriter(BaseRewriter):
    """Rewriter local: gemma3:12b via Ollama (reaproveita o modelo do analyst).

    is_fallback distingue o papel na auditoria: True (default) quando é
    contingência do timeout do gpt-4o; False quando é o rewriter PRINCIPAL
    (LLM_MODEL_PAID=false).
    """

    provider = "ollama"
    model = _OLLAMA_ANALYST_MODEL

    def __init__(self, is_fallback: bool = True):
        self.is_fallback = is_fallback

    def rewrite(self, graph: dict) -> dict:
        from ollama import Client

        client = Client(host=os.environ.get("OLLAMA_HOST", _OLLAMA_DEFAULT_HOST))
        system_prompt, user_message = build_rewriter_prompt(graph)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        with llm_audit.audit_analysis(
            self.provider, self.model, self.is_fallback
        ) as audit:
            # temperature levemente acima de 0: é uma síntese em prosa, não uma
            # extração determinística; um pouco de variação lê melhor.
            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.2, "num_ctx": _OLLAMA_NUM_CTX},
            )
            narrative = (response.message.content or "").strip()
            audit["narrative_chars"] = len(narrative)
        return {**graph, "narrative_summary": narrative}


def load_openai_rewriter() -> OpenAIRewriter:
    """Rewriter pago (gpt-4o). Principal quando LLM_MODEL_PAID=true (default)."""
    return OpenAIRewriter()


def load_ollama_rewriter(is_fallback: bool = True) -> OllamaRewriter:
    """Rewriter local (gemma3:12b via Ollama).

    is_fallback=True (default): contingência no timeout do gpt-4o.
    is_fallback=False: principal (LLM_MODEL_PAID=false).
    """
    return OllamaRewriter(is_fallback=is_fallback)


def load_primary_rewriter() -> BaseRewriter:
    """Rewriter principal da etapa de enriquecimento, conforme LLM_MODEL_PAID.

    Espelha load_primary_analyst(): true (default) usa gpt-4o (o timeout aciona
    o fallback Ollama em main.py); false usa gemma3:12b direto, sem exigir
    OPENAI_API_KEY.
    """
    if _LLM_MODEL_PAID:
        return load_openai_rewriter()
    return load_ollama_rewriter(is_fallback=False)
