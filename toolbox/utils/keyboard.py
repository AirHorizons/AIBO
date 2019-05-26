"""
키보드 이벤트 검출 모듈
"""

__author__ = '박현수(hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import platform
import time

from IPython import embed

# 현재는 esc만 사용
keymap = dict(esc=27)

if platform.system() == 'Windows':
    from msvcrt import kbhit
    from msvcrt import getch

    def event(key):
        if kbhit() and ord(getch()) == keymap[key]:
            return True
        return False

elif platform.system() == 'Linux':
    import curses
    import time

    def _event(stdscr):
        stdscr.nodelay(True)
        return stdscr.getch()

    _event._last_check_time = time.time()

    def event(key):
        if time.time() - _event._last_check_time > 3:
            _event._last_check_time = time.time()
            return curses.wrapper(_event) == keymap[key]
        else:
            return False


if __name__ == '__main__':

    # 키보드 모듈 사용 예
    
    from IPython import embed

    lr = 0.1

    while True:
        if event('esc'):
            print('==== enter debug mode ====')
            # 반복문 실행 도중,
            # 아무때나 esc 키를 누르면 embed() 실행
            embed()
        print('test', lr)
        time.sleep(1)
