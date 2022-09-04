@option struct ModelConfig
    name::String = "resnet"
    arch::String = "resnet18"
    pretrained::Bool = false
end

@option struct SchedulerConfig
    name::String = "step"
    cycle_length::Int = 50000
    damp_factor::Float32 = 1.2f0
    lr_step::Vector{Int64} = [100000, 250000, 500000]
    lr_step_decay::Float32 = 0.1f0
end

@option struct OptimizerConfig
    name::String = "adam"
    learning_rate::Float32 = 0.01f0
    nesterov::Bool = false
    momentum::Float32 = 0.0f0
    weight_decay::Float32 = 0.0f0
    scheduler::SchedulerConfig = SchedulerConfig()
end

@option struct TrainConfig
    total_steps::Int = 800000
    evaluate_every::Int = 10000
    resume::String = ""
    evaluate::Bool = false
    checkpoint_dir::String = "checkpoints"
    log_dir::String = "logs"
    expt_subdir::String = ""
    expt_id::String = ""
    print_frequency::Int = 100
end

@option struct DatasetConfig
    data_root::String = ""
    eval_batchsize::Int = 64
    train_batchsize::Int = 64
end

@option struct ExperimentConfig
    seed::Int = 12345
    model::ModelConfig = ModelConfig()
    optimizer::OptimizerConfig = OptimizerConfig()
    train::TrainConfig = TrainConfig()
    dataset::DatasetConfig = DatasetConfig()
end
