module Serialization

using SciMLPublic: @public

using ..Lux: Lux, is_extension_loaded
using LuxCore: LuxCore, AbstractLuxLayer

include("tf_saved_model.jl")

@public export_as_tf_saved_model

end
