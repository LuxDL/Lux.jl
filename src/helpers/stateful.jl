# Previously this file held the StatefulLuxLayer definition. This has been moved to
# LuxCore.jl
import LuxCore: StatefulLuxLayer

function Base.show(io::IO, ::MIME"text/plain", s::StatefulLuxLayer)
    return PrettyPrinting.print_wrapper_model(
        io, "StatefulLuxLayer{$(dynamic(s.fixed_state_type))}", s.model
    )
end
