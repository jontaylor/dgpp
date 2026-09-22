# Optional tool-call response cap

Chat completions accept the DGPP extension `max_tool_calls`, a positive
32-bit integer. Omit it for the existing unlimited behavior. For example,
add `"max_tool_calls": 3` alongside a nonempty `tools` array. The limit is
per generated choice (`n` choices each have their own limit), not per agent
conversation or across HTTP requests. It requires a tool choice other than
`none`; it does not require the model to call a tool.

On GLM, Qwen and MiMo tool formats, generation stops after the requested
number of complete, successfully parsed calls. The last call's arguments
are returned in full, with `finish_reason: "tool_calls"`, identically for
streaming and non-streaming responses. Earlier reasoning and content are
preserved. Malformed or unfinished blocks do not count; existing literal
content fallback and token limits still apply. A token limit reached
before a complete call remains a length-limited response. Scheduler
retirement uses the ordinary journaled stop path; already computed tokens
past the boundary are not exposed or counted in response usage.

DSML tool containers currently reject this option with HTTP 400: their
parser emits several calls together, and imposing a strict cap there
would discard complete calls. Invalid values, missing tools and
`tool_choice: "none"` also return HTTP 400 naming `max_tool_calls`.
