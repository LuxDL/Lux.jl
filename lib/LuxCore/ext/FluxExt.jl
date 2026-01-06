module FluxExt

using Flux: @layer
using LuxCore: StatefulLuxLayer

@layer StatefulLuxLayer trainable=(ps,)

end
