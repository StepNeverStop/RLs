import time
import socket


def check_port_in_use(port, host='127.0.0.1', try_times=10, server_name='server'):
    for i in range(try_times):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((host, int(port)))
            print(f'{i}: port {port} is under used.')
            time.sleep(1)
        except socket.error:
            return
        finally:
            if s:
                s.close()
    else:
        raise Exception(f'Cannot start {server_name} correctly.')
