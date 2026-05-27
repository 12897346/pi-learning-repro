from src.trainers.minimal_train_demo import DemoConfig, run_minimal_demo


if __name__ == "__main__":
    run_minimal_demo(
        config_path="configs/paper_params.yaml",
        # 仅用于链路自检，默认采用超轻量参数，避免高内存占用
        demo_cfg=DemoConfig(device="cpu", steps=2, batch_size_override=1),
    )
