import pstats
import subprocess

cmd = 'python3 -m cProfile -o profiling_results genetic_optimization.py gothic.jpg'
subprocess.call(cmd, shell=True)

subprocess.call('snakeviz profiling_results', shell=True)

# stats = pstats.Stats("profiling_results")
# stats.sort_stats("tottime")
# stats.print_stats(10)
#
# stats.sort_stats("cumtime")
# stats.print_stats(10)