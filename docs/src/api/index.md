# API Reference

```@contents
Pages =  reduce(vcat, map(readdir(".")) do p
    if isfile(p)
        return p
    else
        return map(pp -> isfile(joinpath(p, pp)) ? joinpath(p, pp) : "", readdir(p))
    end
end)
Depth = 5
```
