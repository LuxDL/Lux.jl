using Pkg, Preferences, Test
using Lux, LuxCore, LuxLib

@testset "Preferences" begin
    @testset "set_dispatch_doctor_preferences!" begin
        @test_throws AssertionError Lux.set_dispatch_doctor_preferences!("invalid")
        @test_throws AssertionError Lux.set_dispatch_doctor_preferences!(;
            luxcore="invalid"
        )

        mktempdir() do tmp_dir
            orig_project = Base.active_project()
            Pkg.activate(tmp_dir)
            try
                pref_file = joinpath(tmp_dir, "LocalPreferences.toml")

                # First call: writes instability_check = error into LocalPreferences.toml
                Lux.set_dispatch_doctor_preferences!("error")
                @test isfile(pref_file)
                content = read(pref_file, String)
                @test occursin("instability_check = \"error\"", content)
                @test Preferences.load_preference(LuxCore, "instability_check") == "error"
                @test Preferences.load_preference(LuxLib, "instability_check") == "error"

                # Second call: hits the "already set to error" @info log without rewriting
                mtime_before = mtime(pref_file)
                @test_logs(
                    (:info, r"already set to error"),
                    (:info, r"already set to error"),
                    Lux.set_dispatch_doctor_preferences!("error")
                )
                @test mtime(pref_file) == mtime_before

                # Updating preferences via keyword arguments
                Lux.set_dispatch_doctor_preferences!(; luxcore="warn", luxlib="disable")
                @test Preferences.load_preference(LuxCore, "instability_check") == "warn"
                @test Preferences.load_preference(LuxLib, "instability_check") == "disable"
            finally
                Pkg.activate(orig_project)
            end
        end
    end
end
