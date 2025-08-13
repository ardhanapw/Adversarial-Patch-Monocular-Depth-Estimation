def format_time(time):
    hour, minute, second = time // 3600, (time % 3600) // 60, time % 3600 % 60
    string = str(hour).zfill(2) + ":" + str(minute).zfill(2) + ":" + str(second).zfill(2)
    return string