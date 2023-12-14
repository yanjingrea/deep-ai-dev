import os

#-----------------------------------------------------------------------------------------------------------------------
# colored text

try:
	import colorama
	from colorama import Fore, Style

	colorama.init()

	highlighted = lambda s, color_seq = Fore.BLUE: \
					f"{Style.BRIGHT + color_seq}{s}{Style.RESET_ALL}"

	bright = lambda s: highlighted(s, Fore.WHITE)
	warning = lambda s: highlighted(s, Fore.YELLOW)
	error = lambda s: highlighted(s, Fore.RED)

except:
	highlighted = lambda s, *args: s
	bright = lambda s: s
	warning = lambda s: s
	error = lambda s: s

#----------------------------------------------------------
# progress bar

try:
	from tqdm import tqdm
except:
	print(warning('tqdm is not installed, progress bars will not be shown'))
	tqdm = lambda a: a

#----------------------------------------------------------
# general

clear_screen = lambda: os.system('cls||clear')

if os.name == 'nt':

	import ctypes
	STD_OUTPUT_HANDLE = ctypes.windll.kernel32.GetStdHandle(ctypes.c_long(-11))

	move_cursor = lambda x, y: \
					ctypes.windll.kernel32.SetConsoleCursorPosition \
					(
						STD_OUTPUT_HANDLE,
						ctypes.c_ulong(x + (y << 16))
					)

else:

	move_cursor = lambda x, y: \
					print(f"\033[{y};{x}H", end = None)

# prints a horizontal bar
#
bar = lambda n = 120: print('-'*n)