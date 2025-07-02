module Serialization

using Compat: @compat

using ..Lux: Lux, is_extension_loaded
using LuxCore: LuxCore, AbstractLuxLayer

include("tf_saved_model.jl")

@compat public export_as_tf_saved_model

end
