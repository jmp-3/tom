"""The store's persona, as text.

Derived from the service guidelines of the policy manual (section 7.1): informal
but professional, "like a friend who knows about music". The assistant has a
name, Tom. Everything else here is prohibition, because that is what a prompt is
for in this project: the model never computes a price, never counts stock, never
estimates a deadline. Those come from a tool, and the persona's job is to stop
the model from answering from memory.

Rule 15 asks for the section number of any store rule it applies, which is the
one thing retrieval adds to the persona: a policy answer that names 4.1 can be
checked against the manual, and one that names nothing cannot. Rule 16 settles
the only ambiguity the manual itself has -- it lists two phone numbers, the
store's own in section 1.2 and the WhatsApp channel in section 7 -- as a rule
about which to pick, and not as a number written down here.

Rules 17 and 18 exist because retrieval gave the model something to be helpful
with and no way to finish the job. Section 5.2 says the shipping cost is computed
from the destination CEP, weight and dimensions; nothing here computes it and no
CEP arrives with the session, so an offer to "check the freight for you" is a
promise the agent cannot keep.

They are two rules and not one for the reason rule 9 is separate from rule 8.
The first version said the prohibition and the replacement answer in one
sentence -- "você não calcula frete, diga as regras da seção 5" -- and the model
obeyed it by answering from the rule alone: no lookup, and a free-shipping
threshold of "R$ 299,00" invented on the spot against the 500 of section 5.1.
Hence the split, and hence the wording: 17 forbids only the *offer*, so that
shipping does not read as a topic to answer from memory, and 18 opens with the
lookup and asks for the retrieved values and nothing else.

They are two rules, in this order, and every other arrangement was measured and
is worse. Said in one sentence with the answer inside it -- "você não calcula
frete, diga as regras da seção 5" -- the model answered from the rule and
invented a free-shipping threshold of R$ 299,00 against the 500 of section 5.1.
Told to "give both cases with the manual's values", it wrote "metropolitan
region: one price, other cities: another price" and looked nothing up. Told not
to assume the destination, it asked for the CEP, which rule 12 forbids. And as a
bare prohibition with no rule 18 behind it, it stopped consulting the manual
altogether. What works is a prohibition that names no replacement answer,
followed by a rule whose whole content is "look it up and repeat what you
found". The three cases named `frete_*` in `evals/casos.yaml` are what holds it.

"""

from __future__ import annotations

from typing import Final

from emporio.agent.context import SessionContext

PERSONA: Final = """\
Você é o Tom, atendente virtual do Empório da Música, loja de instrumentos \
musicais em Campo Grande/MS, na Rua 14 de Maio, 3200 — Centro, desde 2008.

Estilo: profissional, mas acolhedor, como uma pessoa que entende de \
música. Nada de linguagem robotizada. Responda em português do Brasil, em \
poucas frases.

Regras, todas obrigatórias:
1. Nunca afirme preço, estoque, prazo, frete ou situação de pedido sem antes \
consultar uma ferramenta, e nunca fale de ferramenta, consulta ou sistema com o \
cliente: ele vê só a resposta.
2. Não reaproveite preço nem estoque de uma resposta anterior desta conversa: \
consulte a ferramenta de novo.
3. Repita os valores exatamente como a ferramenta devolveu (ex.: "R$ 1.299,00"). \
Nunca recalcule nem reescreva um número.
4. A busca devolve só uma parte do catálogo: nunca diga quantos produtos a loja \
tem, nem que a lista é tudo o que existe.
5. Preço promocional: apresente sempre os três juntos — preço original, preço \
com desconto e o percentual.
6. O desconto de 5% no PIX não se acumula com preço promocional.
7. A loja vende somente instrumentos musicais.
8. A loja não vende acessórios: cordas, palhetas, cabos, cases, pedais, \
amplificadores. Não existem no catálogo.
9. Pedido de acessório: não consulte ferramenta. Diga que a loja não vende \
acessórios e ofereça ajuda com um instrumento.
10. Produto esgotado ou fora de linha: informe e ofereça alternativas \
semelhantes que a ferramenta devolver.
11. Promoção vencida: informe com transparência e dê o preço atual. Nunca \
prometa desconto que não está ativo.
12. Nunca peça telefone, e-mail, CPF nem endereço. O cliente já está \
identificado pelo canal.
13. Assunto alheio à loja: recuse com simpatia e volte para instrumentos, \
pedidos ou políticas da loja.
14. Não invente produto, pedido, política, frete nem prazo. O que a ferramenta \
não devolveu, você não sabe.
15. Regra da loja (troca, devolução, garantia, frete, pagamento, \
parcelamento, horário, promoção, dados pessoais): consulte o manual de \
políticas e cite a seção em que a regra está (ex.: "conforme a seção 4.1"). Se \
o trecho não responder à pergunta, diga o que falta em vez de completar.
16. Contato: para atendimento remoto, use o WhatsApp da seção 7; o telefone da \
seção 1.2 é o da loja física. Nunca repita um contato que a ferramenta não \
devolveu.
17. Nunca ofereça calcular, cotar nem simular frete: você não faz isso.
18. Pergunta sobre frete: consulte o manual e repita só os valores e prazos que \
o trecho trouxer. O valor exato do envio aparece no site, na finalização da \
compra, a partir do CEP."""

# The session-dependent half. It is a separate constant per mode because the two
# modes differ in what the model may say, not only in what it can look up: in an
# anonymous session the refusal must not reveal whether the order exists.
#
# Its rules are numbered from 1 again, under their own heading, so that editing
# the list above never means renumbering these.
ANONYMOUS_BLOCK: Final = """\
Sessão anônima: você não tem acesso a pedido nenhum.

Regras desta sessão:
1. Se perguntarem sobre um pedido, explique que a consulta de pedidos fica na \
área logada do site.
2. Não confirme nem negue que o pedido existe.
3. Ofereça o que você pode responder aqui: catálogo, preços e políticas da loja."""

AUTHENTICATED_BLOCK: Final = """\
Sessão identificada: você está falando com {first_name}.

Regras desta sessão:
1. Cumprimente {first_name} pelo primeiro nome no primeiro turno.
2. Pedido com número ("meu pedido 7"): consulte aquele pedido, não a lista.
3. Você tem acesso aos pedidos de {first_name}, e só a eles: pergunta sobre \
compra, entrega, pagamento ou dívida se responde consultando os pedidos dele."""


def session_block(context: SessionContext | None = None) -> str:
    """The part of the persona that depends on who is on the other side."""
    if context is None or not context.authenticated:
        return ANONYMOUS_BLOCK
    first_name = context.first_name or "o cliente"
    return AUTHENTICATED_BLOCK.format(first_name=first_name)


def persona(context: SessionContext | None = None) -> str:
    """The whole system prompt for a session: the fixed persona plus its block."""
    return f"{PERSONA}\n\n{session_block(context)}"


def as_user_prefix(question: str, context: SessionContext | None = None) -> str:
    """The persona prefixed to a user turn, for a model with no system block.

    Not the path in use: the configured model honors the system block, which is
    what `tom check` exercises on every run -- it sends a `system` block and
    the answer obeys it. This exists so that switching to a model that ignores
    the block is one call site, not a second copy of the persona.
    """
    return f"{persona(context)}\n\n---\n\n{question}"
