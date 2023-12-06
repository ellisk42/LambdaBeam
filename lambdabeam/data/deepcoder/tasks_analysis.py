"""Analysis of tasks."""

import collections

from lambdabeam.data.deepcoder import deepcoder_tasks
from lambdabeam.data.deepcoder import solution_weight


HANDWRITTEN_WEIGHT_BUCKETS = [(4, 6), (7, 8), (9, 10), (11, 12), (13, 19)]
SYNTHETIC_WEIGHT_BUCKETS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]

TaskAnalysis = collections.namedtuple(
    'TaskAnalysis', ['task', 'weight', 'has_lambda', 'output_type'])


def analyze_tasks(tasks):
  return [
      TaskAnalysis(
          task=t,
          weight=solution_weight.solution_weight(t.solution),
          has_lambda='lambda' in t.solution,
          output_type=type(t.outputs[0]).__name__)
      for t in tasks
  ]


def pad(text, length):
  if not isinstance(text, str):
    text = str(text)
  if len(text) < length:
    return f'\\phantom{{{"0" * (length - len(text))}}}' + text
  elif len(text) == length:
    return text
  else:
    raise ValueError(f'Text is more than length {length}: {text}')


def main():
  handwritten = analyze_tasks(deepcoder_tasks.HANDWRITTEN_TASKS)
  synthetic = analyze_tasks(deepcoder_tasks.SYNTHETIC_TASKS)
  weights = range(3, 20)

  lines = {w: f'{pad(w, 2)}' for w in weights}
  total_hw_num = total_hw_lambda = total_hw_int = 0
  total_syn_num = total_syn_lambda = total_syn_int = 0
  space = r'\hspace{0.3em}'

  for task_list, is_hw in ((handwritten, True), (synthetic, False)):
    for weight in weights:
      tasks = [t for t in task_list if t.weight == weight]
      num = len(tasks)
      if is_hw:
        total_hw_num += num
      else:
        total_syn_num += num

      if num:
        num_lambda = len([t for t in tasks if t.has_lambda])
        num_non_lambda = len([t for t in tasks if not t.has_lambda])
        assert num_lambda + num_non_lambda == num
        if is_hw:
          total_hw_lambda += num_lambda
        else:
          total_syn_lambda += num_lambda
        lambda_percent = pad(f'{num_lambda * 100 / num:.0f}\\%',
                             5 if is_hw else 4)
        num_lambda = pad(num_lambda, 2)
        lambda_text = f'{num_lambda} {space} {lambda_percent}'

        num_int = len([t for t in tasks if t.output_type == 'int'])
        num_list = len([t for t in tasks if t.output_type == 'list'])
        assert num_int + num_list == num
        if is_hw:
          total_hw_int += num_int
        else:
          total_syn_int += num_int
        int_percent = pad(f'{num_int * 100 / num:.0f}\\%', 4)
        num_int = pad(num_int, 2)
        int_text = f'{num_int} {space} {int_percent}'
      else:
        lambda_text = int_text = '--'

      lines[weight] += f' & {pad(num, 3)} & {lambda_text} & {int_text}'

  for weight in weights:
    print(lines[weight] + ' \\\\')
  print('\\midrule')

  assert total_hw_num == 100 and total_syn_num == 100
  total_hw_lambda_text = (
      f'{pad(total_hw_lambda, 2)} {space} {pad(total_hw_lambda, 3)}\\%')
  total_hw_int_text = (
      f'{pad(total_hw_int, 2)} {space} {pad(total_hw_int, 2)}\\%')
  total_syn_lambda_text = (
      f'{pad(total_syn_lambda, 2)} {space} {pad(total_syn_lambda, 2)}\\%')
  total_syn_int_text = (
      f'{pad(total_syn_int, 2)} {space} {pad(total_syn_int, 2)}\\%')
  total_line = 'Total'
  for part in [
      total_hw_num, total_hw_lambda_text, total_hw_int_text,
      total_syn_num, total_syn_lambda_text, total_syn_int_text]:
    total_line += ' & ' + str(part)
  print(total_line + ' \\\\')

if __name__ == "__main__":
  main()
