"""The session context: what the channel knows before the conversation starts.

The chat is simulated as if it lived inside the logged-in area of the store's
site, so the customer's identity arrives with the session and is never asked for
in conversation. `create_agent(context_schema=SessionContext)` is what carries it
to the tools: a tool that declares `runtime: ToolRuntime` reads
`runtime.context.customer_id`, and the parameter is stripped from the schema the
model sees. The model therefore has no way to fill `customer_id` in -- the
privacy rule of the data layer is enforced by the framework, not by a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from emporio.data import queries
from emporio.data.loader import Dataset


@dataclass(frozen=True)
class SessionContext:
    """Identity for one chat session, or nothing at all when it is anonymous.

    Frozen: the session is decided once, before the first turn, and a tool that
    could rewrite `customer_id` mid-run would be the same hole as letting the
    model fill it.
    """

    customer_id: int | None = None
    first_name: str | None = None
    city: str | None = None

    @property
    def authenticated(self) -> bool:
        return self.customer_id is not None


# The anonymous session, spelled out so that no caller has to decide what
# "no context" means. Public catalog information only.
ANONYMOUS: Final = SessionContext()


def for_customer(customer_id: int, *, dataset: Dataset | None = None) -> SessionContext | None:
    """Build the session of an identified customer. None when the id is unknown.

    Called once per session, by the CLI. `load_customer` is the one query that is
    not a tool: the name and the city are session data, not something the model
    can look up, and no phone or e-mail is projected (policy manual, section 9).
    """
    row: dict[str, Any] | None = queries.load_customer(customer_id, dataset=dataset)
    if row is None:
        return None
    # The persona greets by first name, which is all of the name it ever needs.
    first_name = str(row["name"]).split()[0]
    return SessionContext(
        customer_id=int(row["customer_id"]), first_name=first_name, city=row["city"]
    )
