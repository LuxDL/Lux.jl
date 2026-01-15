module FluxExt

using Flux: @layer
using LuxCore: StatefulLuxLayer

@layer :ignore StatefulLuxLayer trainable = (ps,)

end
