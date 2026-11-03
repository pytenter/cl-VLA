"""Run one NEW ordinary-LoRA plus fixed-prototype-replay stage."""
import argparse
import dataclasses
import json
import logging
import sys
import time


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['plain_prototype'], required=True)
    parser.add_argument('--stage', choices=list('ABCD'), required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--weights')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from experiment_config import make_config, ROOT
    from libero_integration import install_hooks
    install_hooks()
    config = make_config(args.group, args.stage, steps=args.steps, batch_size=args.batch_size,
                         resume=args.resume, smoke=args.smoke, weights=args.weights)
    folder = ROOT/'reports/configurations'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'{config.name}.json').write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2))
    if config.rehearsal.enabled:
        from plain_replay_training import install_training_fixes
        trainer = install_training_fixes(metrics_path=ROOT/'logs'/f'{config.name}.metrics.jsonl').main
        runtime = 'native_replay_loop_with_ordinary_compute_loss_and_initializer'
    else:
        from scripts.train import main as trainer
        runtime = 'native_ordinary_train'
    started = time.monotonic()
    trainer(config)
    report = {'group': args.group, 'stage': args.stage, 'steps': config.num_train_steps,
              'batch_size': config.batch_size, 'seconds': time.monotonic() - started,
              'checkpoint_dir': str(config.checkpoint_dir), 'smoke': args.smoke,
              'runtime': runtime, 'replay_buffer_dirs': config.rehearsal.buffer_dirs,
              'replay_buffer_policy': 'Read-only original CL-prototype buffers; fixed selector content for architecture comparison'}
    (ROOT/'reports'/f'{config.name}_complete.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
