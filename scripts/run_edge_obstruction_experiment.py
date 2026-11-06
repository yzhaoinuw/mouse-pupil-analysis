"""Run the matched BCE edge-obstruction comparison."""

if __package__:
    from .run_augmentation_experiment import main
else:
    from run_augmentation_experiment import main


if __name__ == "__main__":
    main("edge")
