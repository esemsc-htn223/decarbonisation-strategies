from typing import Literal
from pydantic import BaseModel, Field, create_model
from decarb.terms import Term


# ---------------------------------------------------------------------------------------------
#                                               V1
# ---------------------------------------------------------------------------------------------

class QuoteEvidence(BaseModel):
    quote: str = Field(description="Direct quote from the source text")
    justification: str = Field(description="Explanation of why this quote is relevant as evidence")

class Commitment(BaseModel):
    commitment: str = Field(description="A commitment made by the company towards the technology")
    quote: str = Field(description="Direct quote from the source text evidencing the commitment")
    justification: str = Field(description="Explanation of how the quote evidences the commitment")
    verifiable: bool = Field(description="True if the commitment includes specific, measurable targets. False if the commitment is aspirational or qualitative only.")

class PromptResponseV1(BaseModel):
    # Structured decoding emits fields in schema order, so each question's quotes are declared
    # before its answer: the model gathers its evidence, then scores against it.

    # Question 1: "How positive or negative is the overall sentiment towards {term.prompt_alias}?"
    q1_sentiment_quotes: list[QuoteEvidence] = Field(
        default=[],
        description="Quotes reflecting sentiment towards the technology and a justification for their inclusion. q1_sentiment must use these quotes to assign a sentiment score."
    )
    q1_sentiment: float | None = Field(
        default=None,
        description="Question 1: Sentiment towards the technology, on a scale from -1 (totally negative) to 1 (totally positive), based on the quotes given in q1_sentiment_quotes",
        ge=-1.0, le=1.0
    )
    q1_framing: Literal["opportunity", "neutral", "risk"] = Field(
        default="neutral",
        description="Question 1b: Whether the extracts frame the technology as a commercial opportunity for the company, as a risk or threat to it, or neither. Based on the quotes given in q1_sentiment_quotes."
    )

    # Question 2: "How mature is {term.prompt_alias} framed to be, in terms of Techonology Readiness Levels?"
    q2_maturity_quotes: list[QuoteEvidence] = Field(
        default=[],
        description="Quotes reflecting the Technology Readiness Level (TRL) of the technology and a justification for their inclusion. q2_maturity must use these quotes to assign a TRL."
    )
    q2_maturity: int | None = Field(
        default=None,
        description="Question 2: The Technology Readiness Level (TRL) of the technology as framed in the extracts, 1-9, based on the quotes given in q2_maturity_quotes. Definitions provided in the prompt. Null if insufficient information is available to assign a TRL.",
        ge=1, le=9
    )

    # Question 3: "What stage of commercialisation is {term.prompt_alias} at within the company's operations?"
    q3_stage_quotes: list[QuoteEvidence] = Field(
        default=[],
        description="Quotes reflecting the technology's stage of commercialisation within the company's operations and a justification for their inclusion. q3_stage must use these quotes to assign a stage."
    )
    q3_stage: Literal["research and development", "pilot", "demonstration", "commercial-deployment", "phasing-out", "unclear"] = Field(
        default="unclear",
        description="Question 3: The technology's stage of commercialisation within the company's operations, based on the quotes given in q3_stage_quotes."
    )

    # Question 4: "What commitments does the company make towards {term.prompt_alias}?"
    q4_commitments: list[Commitment] = Field(
        default=[],
        description="Question 4: Commitments made towards the technology."
    )


# ---------------------------------------------------------------------------------------------
#                                               V2
# ---------------------------------------------------------------------------------------------

drivers = Literal[  # drivers of change or decisions
    'financial_returns', 'market_conditions', 'customer_demand', 'policy_or_regulation',
    'technology_development', 'partners_or_financing', 'energy_security', 'sustainability',
    'stakeholder_pressure', 'other',
]
DRIVER_DEFINITIONS = (
    "financial_returns: profitability, hurdle rates, shareholder returns. "
    "market_conditions: commodity or power prices, costs, interest rates, supply chain. "
    "customer_demand: demand from customers or offtakers. "
    "policy_or_regulation: subsidies, tax credits, mandates, carbon pricing, permits. "
    "technology_development: cost reduction, maturity or performance of the technology. "
    "partners_or_financing: participation of partners or availability of finance. "
    "energy_security: security or affordability of energy supply. "
    "sustainability: climate, emissions or wider environmental goals. "
    "stakeholder_pressure: investors, NGOs, litigation, public opinion. "
    "other: none of the above."
)
CITED_EVENT_EXAMPLES = [
    ["Inflation Reduction Act"],
    ["war in Ukraine", "European energy crisis"],
    ["COVID-19 pandemic"],
    ["Hurricane Katrina"],
    ["appointment of our new Chief Executive"],
    ["reorganisation into four business segments"],
    [],
]


class Quantity(BaseModel):
    value: float = Field(
        description=(
            "The figure exactly as written, without converting its scale ('1.5 million tonnes' -> 1.5). "
            "If a range with a lower and upper bound is given, this is the lower bound."
        )
    )
    value_high: float | None = Field(
        description="Upper bound if a range is given ('$1-2 billion' -> 2). None if not a range."
    )
    bound: Literal['exact', 'approximate', 'at_least', 'up_to'] = Field(
        description=(
            "approximate: 'around', 'about', 'c.'. at_least: 'more than', 'over', 'at least'. "
            "up_to: 'up to', 'as much as', 'less than'. exact: otherwise."
        )
    )
    unit: str = Field(
        description=(
            "Unit as written, including scale. Time frames such as 'per year' should be included in the period field rather than here, "
            "unless it is part of a single unit (e.g., 'Mtpa')."
        ),
        examples=['GW', 'MW', 'Mtpa', '$ billion', '€ million', '% of capex', 'MtCO2e', 'charge points'],
    )
    metric: Literal[
        'capacity', 'production_or_sales', 'capital_expenditure', 'other_spending',
        'financial_value', 'emissions', 'count', 'other',
    ] = Field(
        description=(
            "What is measured by the figure. capacity: installed or planned capacity. production_or_sales: "
            "volumes produced, sold or delivered. capital_expenditure: capex. other_spending: opex, R&D "
            "or procurement spend. financial_value: acquisition price, impairment, revenue or earnings. "
            "emissions: emissions reduced, avoided, captured or stored. count: number of sites, stations, "
            "projects, vehicles, etc. For percentages, give the metric of the whole ('30% of capex' -> "
            "capital_expenditure)."
        )
    )
    period: Literal['annual', 'sub-annual', 'cumulative', 'level'] | None = Field(
        description=(
            "annual: a rate per year ('$1 billion a year'). sub-annual: a rate over a shorter period (e.g., '$1 billion per quarter'). cumulative: a total over a span ('$10 "
            "billion to 2030'). level: a stock at a point in time ('2 GW installed'). None if unclear."
        )
    )
    basis: Literal['gross', 'net'] | None = Field(
        description=(
            "gross: the whole project or asset (100%). net: the company's equity share. None if not stated."
        )
    )


class Action(BaseModel):
    # --- Evidence and identity (generated first, so the labels below are based on the evidence) ---
    quotes: list[str] = Field(
        min_length=1,
        description=(
            "Verbatim excerpts from the source text copied character-for-character that together demonstrate "
            "this action. Every attribute below is based on only these extracts. Use the shortest "
            "spans that suffice, without abbreviating or paraphrasing, including any reasoning or justification. "
            "If the same action appears in several places, record it once and include each excerpt."
        ),
    )
    summary: str = Field(
        description=(
            "One plain sentence restating the action in the form: what is done, where, how much, "
            "by when. Describe the same action the same way however the text phrases it."
        ),
        examples=["Build a 200 MW electrolyser at the Rotterdam refinery, start-up targeted for 2025."],
    )
    project_name: str | None = Field(
        description="Name of the specific project, asset, venture or programme if named. None for unnamed or portfolio-level actions.",
        examples=["Holland Hydrogen 1", None],
    )

    # --- What the action is ---
    action_type: Literal[
        'target', 'investment', 'divestment', 'research', 'partnership',
        'trade', 'organisational', 'advocacy', 'other',
    ] = Field(
        description=(
            "target: a goal or ambition not tied to one project ('X GW by 2030'). "
            "investment: anything that puts capital into the technology (e.g., building, developing or operating assets, acquisitions, equity stakes). "
            "divestment: anything that removes capital from the technology (e.g., selling, exiting, closing, impairing, writing down). "
            "research: R&D, pilots and demonstrations. "
            "partnership: Memorandum of understanding, joint venture formation or collaboration with no stated capital. "
            "trade: buying (procurement, offtake, PPAs) or selling products and services to customers. "
            "organisational: business units, teams, leadership roles, governance, incentives. "
            "advocacy: public policy positions, lobbying, industry associations."
            "other: none of the above."
        )
    )
    technology_scope: Literal['specific', 'bundled'] = Field(
        description=(
            "'bundled' if the action covers this technology together with others (e.g. a 'low-carbon' "
            "capex budget), so its figures cannot be attributed to this technology alone."
        )
    )
    # Narrowed to a per-technology Literal by build_extraction_model().
    technology_variants: list[str] = Field(
        description="Sub-types of the technology this action concerns; empty if not stated."
    )
    company_role: list[Literal['user', 'supplier', 'investor', 'technology_developer']] = Field(
        description=(
            "The company's relationship to the technology in this action. "
            "user: uses the technology or its output in its existing businesses (e.g. solar powering oilfields, hydrogen used in its "
            "refineries, capture on its own LNG plants). "
            "supplier: sells the technology, its output or a related service to others. "
            "investor: holds a financial stake without operating. "
            "technology_developer: develops technology to commercialise or license. "
            "Empty if unclear."
        )
    )

    # --- How firm it is ---
    stage: Literal['exploring', 'proposed', 'committed', 'underway', 'achieved'] = Field(
        description=(
            "Real-world progress, in increasing order. "
            "exploring: evaluating, studying, screening opportunities. "
            "proposed: announced plan or target with no binding decision. "
            "committed: final investment decision, signed contract or approved budget. "
            "underway: construction or execution in progress, or progress reported against a target. "
            "achieved: operating, completed, or target met."
        )
    )
    wording_strength: Literal['definite', 'intended', 'aspirational', 'tentative'] = Field(
        description=(
            "Strength of the wording that governs the action, independent of stage. "
            "definite: will/have/are. "
            "intended: plan/intend/expect. "
            "aspirational: aim/aspire/ambition/seek. "
            "tentative: may/could/consider/potential."
        )
    )
    conditions: list[drivers] = Field(
        description=(
            "Drivers the text explicitly makes this action depend on ('subject to', 'if', 'provided "
            "that'). Ignore boilerplate forward-looking-statement disclaimers. Empty if unconditional. "
            + DRIVER_DEFINITIONS
        )
    )
    status: Literal['on_track', 'expanded', 'delayed_or_paused', 'under_review', 'reduced', 'cancelled_or_exited'] | None = Field(
        description=(
            "Change to this action that the text explicitly reports. "
            "on_track: progressing as planned. "
            "expanded: scaled up or accelerated. "
            "delayed_or_paused: pushed back or put on hold. "
            "under_review: being reassessed with outcome unknown. "
            "reduced: scaled back. "
            "cancelled_or_exited: stopped, abandoned or exited. "
            "None if no change reported - do not infer one."
        )
    )

    # --- Why (needed for attribution) ---
    reasons: list[drivers] = Field(
        description=(
            "Motivations or explanations the text explicitly gives for the action or its status. Empty if "
            "none stated. " + DRIVER_DEFINITIONS
        )
    )
    cited_events: list[str] = Field(
        description=(
            "Specific named events, external or internal to the company, that the text connects to this "
            "action, worded as in the text. Empty if none."
        ),
        examples=CITED_EVENT_EXAMPLES,
    )

    # --- Quantity ---
    quantities: list[Quantity] = Field(
        description="Every figure attached to the action (capacity, spend, volume, etc.), excluding dates. Empty if no figures are stated."
    )

    # --- Who and where ---
    partners: list[str] = Field(description="Named partner organisations. Empty if none.")
    location: str | None = Field(
        description="Country, or region if no country is given, where the action takes place; 'global' if global, and None if no location stated."
    )

    # --- When ---
    start_year: int | None = Field(
        ge=1900, le=2150,
        description=(
            "Year the action began or is scheduled to begin (decision, construction start, launch). Only if "
            "stated or directly derivable from the text and the report year; never guess."
        ),
    )
    start_month: int | None = Field(ge=1, le=12, description="Month of start_year, only if explicitly stated.")
    target_year: int | None = Field(
        ge=1900, le=2150,
        description="Year by which the action is due to be completed or the target met ('by 2030' -> 2030).",
    )
    timing_text: str | None = Field(
        description="Timing phrase(s) copied verbatim from the text.",
        examples=["by the end of the decade", "in the second half of 2025", None],
    )


class TechnologyStance(BaseModel):
    quotes: list[str] = Field(
        description=(
            "Verbatim excerpts, copied character-for-character, of general statements about the "
            "technology's role, prospects or the company's strategy towards it, as opposed to specific "
            "actions. Empty if there are none."
        )
    )
    off_topic_extracts: list[str] = Field(
        description=(
            "List of indexes i taken from the header ('# Extract ... (section i / ...)') of any supplied extracts that do not discuss the technology at all "
            "(e.g. matched on an unrelated use of a keyword). Empty if all are relevant."
        )
    )
    technology_outlook: Literal['negative', 'mixed', 'neutral', 'positive'] | None = Field(
        description=(
            "What the company says about the technology's future role in the energy system or economy, "
            "regardless of its own involvement. None if it expresses no view."
        )
    )
    strategic_role: Literal['not_pursued', 'peripheral', 'option', 'growth', 'core'] | None = Field(
        description=(
            "The technology's place in the company's own strategy, judged from the text as a whole, "
            "including the actions listed. In increasing order. not_pursued: explicitly not a focus, or "
            "exited. peripheral: mentioned in passing or in a minor supporting role. option: small, early "
            "positioning to keep options open. growth: a stated priority for expansion or significant "
            "capital. core: central to the current business and earnings. None if there is no basis to judge."
        )
    )
    strategy_change: Literal['increased_emphasis', 'reduced_emphasis', 'redirected'] | None = Field(
        description=(
            "Explicit statement that the company's strategy towards this technology has changed. "
            "redirected: shifted between sub-types, markets or business models without a clear increase "
            "or decrease. None if no change is stated - do not infer one."
        )
    )
    change_reasons: list[drivers] = Field(
        description="Explanations given for strategy_change. Empty if none. " + DRIVER_DEFINITIONS
    )
    cited_events: list[str] = Field(
        description="Specific named events the text connects to the company's stance or its change, worded as in the text. Empty if none.",
        examples=CITED_EVENT_EXAMPLES,
    )


class PromptResponseV2(BaseModel):
    actions: list[Action] = Field(
        description="Every distinct action the company reports or announces concerning the technology. Empty if none."
    )
    stance: TechnologyStance

def build_v2_schema(term:Term) -> type[PromptResponseV2]:
    """
    Return an PromptResponseV2 subclass with Action.technology_variants limited 
    to the variants defined in the given Term.

    Parameters
    ----------
    term : Term
        A Term object containing the technology's variants and other metadata.

    Returns
    -------
    type[PromptResponseV2]
        A subclass of PromptResponseV2 with Action.technology_variants constrained to the term's variants.
    """
    variants = {**term.variants, 'other': "a sub-type not listed above"}
    variant_literal = Literal[tuple(variants)]
    definitions = " ".join(f"{v}: {desc}." for v, desc in variants.items())
    term_identifier = term.prompt_fname.replace(".txt", "")
    action_model = create_model(
        f'Action_{term_identifier}',
        __base__=Action,
        technology_variants=(
            list[variant_literal],
            Field(description=(
                "Sub-types of the technology this action concerns, using the most specific that applies; "
                f"empty if not stated. {definitions}"
            )),
        ),
    )
    return create_model(
        f'PromptResponseV2_{term_identifier}',
        __base__=PromptResponseV2,
        actions=(
            list[action_model],
            Field(description=PromptResponseV2.model_fields['actions'].description),
        ),
    )






RESPONSE_SCHEMAS:dict[int, type[BaseModel]] = {
    1: PromptResponseV1,
    2: build_v2_schema,  # built per term
}