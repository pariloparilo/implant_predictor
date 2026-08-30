import argparse
import sys

from .audit import audit
from .common import load_config
from .split import split_draft
from .study import freeze_split


def main():
    parser = argparse.ArgumentParser(description='폐쇄망 임플란트 데이터 점검·분할·학습·평가')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'split-draft', 'freeze-split', 'train', 'evaluate'):
        command = sub.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--run-id', required=True)
        if name == 'audit':
            command.add_argument('--limit', type=int)
        elif name == 'split-draft':
            command.add_argument('--audit-dir', required=True)
        elif name == 'freeze-split':
            command.add_argument('--split-dir', required=True)
            command.add_argument('--review-note', required=True)
        else:
            command.add_argument('--frozen-dir', required=True)
            if name == 'train':
                command.add_argument('--resume')
                command.add_argument('--synthetic-smoke', action='store_true',
                                     help='합성 데이터 실행 시험만: 사전학습 없이 무작위 초기화')
            else:
                command.add_argument('--checkpoint', required=True)
                command.add_argument('--partition', choices=['dev', 'test'], default='dev')
                command.add_argument('--final-test-note')
    command = sub.add_parser('plot')
    command.add_argument('--evaluation-dir', required=True)
    command.add_argument('--training-dir')
    args = parser.parse_args()
    try:
        if args.command == 'plot':
            from .figures import plot_results
            out = plot_results(args.evaluation_dir, args.training_dir)
        else:
            config = load_config(args.config)
            if args.command == 'audit':
                out = audit(config, args.run_id, args.limit)
            elif args.command == 'split-draft':
                out = split_draft(config, args.audit_dir, args.run_id)
            elif args.command == 'freeze-split':
                out = freeze_split(config, args.split_dir, args.run_id, args.review_note)
            elif args.command == 'train':
                from .learning import train
                out = train(config, args.frozen_dir, args.run_id, args.resume, args.synthetic_smoke)
            else:
                from .learning import evaluate
                out = evaluate(config, args.frozen_dir, args.checkpoint, args.partition,
                               args.run_id, args.final_test_note)
        print(f'완료: {out}')
        print('모든 결과는 안심존 내부 전용입니다. 반출·공유에는 기관 검토가 필요합니다.')
    except ImportError as exc:
        print(f'중단: 필요한 패키지가 없습니다. 승인된 설치 목록을 확인하세요: {exc}', file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f'중단: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
