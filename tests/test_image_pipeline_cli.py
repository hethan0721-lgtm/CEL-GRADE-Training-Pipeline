"""
CLI / argparse behavior tests for M0_IMG (image_pipeline.src.main).

None of these tests enter real training, evaluation, or prediction --
failure paths short-circuit inside main() before any heavy work starts,
and the one "valid args" test monkeypatches train_mode() to a no-op stub
so the CLI plumbing is exercised without running the real pipeline.
"""

import sys

import pytest

import image_pipeline.src.main as main_module
from image_pipeline.src.config import Config
from image_pipeline.src.main import build_arg_parser, main


@pytest.fixture(autouse=True)
def _protect_config_globals(monkeypatch):
    """
    main() mutates Config.RANDOM_STATE / MODEL_SAVE_DIR / RESULTS_SAVE_DIR /
    DEVICE as a side effect of parsing --seed/--output-dir/--device. Guard
    these class attributes so each test in this file starts and ends with
    the pristine Config values, regardless of test execution order.
    """
    for attr in ('RANDOM_STATE', 'MODEL_SAVE_DIR', 'RESULTS_SAVE_DIR', 'DEVICE'):
        monkeypatch.setattr(Config, attr, getattr(Config, attr))
    yield


def test_help_lists_all_required_flags(capsys, monkeypatch):
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(['--help'])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    for flag in ('--mode', '--data-dir', '--output-dir', '--checkpoint',
                 '--image', '--device', '--seed'):
        assert flag in captured.out


def test_train_missing_data_dir_fails_clearly(monkeypatch, tmp_path):
    output_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'train', '--output-dir', str(output_dir)
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    # No output directories should have been created before the failure.
    assert not (output_dir / 'models').exists()
    assert not (output_dir / 'results').exists()


def test_evaluate_missing_data_dir_fails_clearly(monkeypatch, tmp_path):
    output_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'evaluate', '--output-dir', str(output_dir)
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_predict_missing_image_fails_clearly(monkeypatch, tmp_path):
    output_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'predict', '--output-dir', str(output_dir)
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_train_nonexistent_data_dir_fails_clearly(monkeypatch, tmp_path):
    missing_data_dir = tmp_path / 'does_not_exist'
    output_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'train',
        '--data-dir', str(missing_data_dir),
        '--output-dir', str(output_dir),
    ])
    with pytest.raises(FileNotFoundError):
        main()


def test_evaluate_nonexistent_data_dir_fails_clearly(monkeypatch, tmp_path):
    missing_data_dir = tmp_path / 'does_not_exist'
    output_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'evaluate',
        '--data-dir', str(missing_data_dir),
        '--output-dir', str(output_dir),
    ])
    with pytest.raises(FileNotFoundError):
        main()


def test_predict_nonexistent_image_fails_clearly(monkeypatch, tmp_path):
    missing_image = tmp_path / 'does_not_exist.png'
    output_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'predict',
        '--image', str(missing_image),
        '--output-dir', str(output_dir),
    ])
    with pytest.raises(FileNotFoundError):
        main()


def test_device_cuda_fails_loudly_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'train', '--device', 'cuda',
        '--data-dir', str(tmp_path / 'irrelevant'),
        '--output-dir', str(tmp_path / 'out'),
    ])
    with pytest.raises(RuntimeError, match='cuda'):
        main()


def test_device_auto_falls_back_to_cpu_when_cuda_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module.torch.cuda, 'is_available', lambda: False)
    output_dir = tmp_path / 'out'
    calls = []
    monkeypatch.setattr(main_module, 'train_mode',
                         lambda *a, **kw: calls.append((a, kw)))
    data_dir = tmp_path / 'data'
    for cls in ('0', '1', '2'):
        (data_dir / cls).mkdir(parents=True)

    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'train', '--device', 'auto',
        '--data-dir', str(data_dir),
        '--output-dir', str(output_dir),
    ])
    main()
    assert len(calls) == 1
    device_arg = calls[0][0][0]
    assert device_arg.type == 'cpu'


def test_valid_train_args_dispatch_to_train_mode_without_real_training(monkeypatch, tmp_path):
    data_dir = tmp_path / 'data'
    for cls in ('0', '1', '2'):
        (data_dir / cls).mkdir(parents=True)
    output_dir = tmp_path / 'out'

    calls = []

    def fake_train_mode(device, data_dir_arg, models_dir, results_dir):
        calls.append((device, data_dir_arg, models_dir, results_dir))

    monkeypatch.setattr(main_module, 'train_mode', fake_train_mode)
    monkeypatch.setattr(sys, 'argv', [
        'prog', '--mode', 'train',
        '--data-dir', str(data_dir),
        '--output-dir', str(output_dir),
        '--device', 'cpu',
    ])

    main()

    assert len(calls) == 1
    device, data_dir_arg, models_dir, results_dir = calls[0]
    assert device.type == 'cpu'
    assert data_dir_arg == data_dir.resolve()
    assert models_dir == output_dir.resolve() / 'models'
    assert results_dir == output_dir.resolve() / 'results'
    # main() itself creates these dirs before dispatching to train_mode;
    # they live under pytest's tmp_path, never under the real repository.
    assert models_dir.is_dir()
    assert results_dir.is_dir()
