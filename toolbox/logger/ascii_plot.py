
__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import os
import logging
import platform


def draw_line(values, title=None, print_out=False):
    """
    gnuplot을 이용해서 ascii 문자로 꺽은선 그래프를 그려줌
    GNU plot을 별도로 설치해야 함
    - Windows (C:/Program Files/gnuplot/bin/gnuplot.exe)
    - Linux (/usr/bin/gnuplot)
    설치가 되어있지 않으면 경고 메시지만 출력함

    :param values: value := [(x1, y1), (x2, y2), ...]
    :param title: str, 그래프 위에 표시할 문자열
    :param print_out: bool, True일 때는 반환하는 것과 별개로 그래프를 print 함
    :returns: str, ascii 문자로 만든 그래프
    """
    import subprocess
    import platform
    if platform.system().lower() == 'windows':
        path = 'C:/Program Files/gnuplot/bin/gnuplot.exe'
    else:
        path = "/usr/bin/gnuplot"

    try:
        if 'linux' in platform.system().lower():
            width = max(50, os.get_terminal_size().columns)
            height = max(25, os.get_terminal_size().lines / 4)
        else:  # windows
            width = max(50, os.get_terminal_size().columns - 2)
            height = max(25, os.get_terminal_size().lines / 4)

        gnuplot = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        input_string = "set term dumb {} {}\n".format(width, height)
        input_string += "set key off\n"

        input_string += "plot "
        input_string += ', '.join(
            ["'-' using 1:2 title 'Line1' with linespoints" for _ in values])
        input_string += '\n'

        for xy in values:
            xs, ys = zip(*xy)
            for i, j in zip(xs, ys):
                input_string += "%f %f\n" % (i, j)
            input_string += "e\n"

        output_string, error_msg = gnuplot.communicate(input_string)

        if title is not None:
            title = '** {} **\n'.format(title.title())
            if 'linux' in platform.system().lower():
                # 불필요한 공백 줄 제거
                output_string = title + output_string[2 * width:-(width+1)]
            else:  # windows
                output_string = title + output_string[width:-(width+1)]

        if print_out:
            print(output_string)

        return output_string
    except FileNotFoundError:
        logging.warning("Can't find gnuplot")
        return ''


if __name__ == '__main__':

    out = draw_line([[(1, 1), (2, 2)]], title='Test', print_out=True)
    import IPython
    IPython.embed()
