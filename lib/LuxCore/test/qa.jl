using LuxCore, Test, ExplicitImports, Aqua

@testset "Quality Assurance" begin
    Aqua.test_all(LuxCore)
    # The Functors extension intentionally reaches LuxCore's internal extension hooks.
    ExplicitImports.test_explicit_imports(
        LuxCore;
        all_qualified_accesses_are_public=(;
            ignore=(
                :Internal,
                :StatefulLuxLayerImpl,
                :fleaves_impl,
                :fmap_impl,
                :fmap_with_path_impl,
                :is_extension_loaded,
                :isleaf_impl,
                :setfield,
            ),
        ),
    )
end
