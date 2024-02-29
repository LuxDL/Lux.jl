abstract type AbstractFrameworkAdaptor end

abstract type AbstractToLuxAdaptor <: AbstractFrameworkAdaptor end
abstract type AbstractFromLuxAdaptor <: AbstractFrameworkAdaptor end

(adaptor::AbstractFrameworkAdaptor)(x) = adapt(adaptor, x)
