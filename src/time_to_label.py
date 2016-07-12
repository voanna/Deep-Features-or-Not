# For directories containing images named like 2014-05-14_09-47-00.jpg,
# convert day to season
import os
import dateutil.parser
import glob
import ipdb 

def season(timetuple):
    '''
    Extracts the season from dateutil timetuple object, and outputs season labels in 0, 1, 2, 3 starting from winter
    >>> import dateutil
    >>> timetuple = dateutil.parser.parse('2013-04-07_17-02-21', fuzzy=True).timetuple()
    >>> season(timetuple)
    1

    >>> timetuple = dateutil.parser.parse('2013-01-07_17-12-21', fuzzy=True).timetuple()
    >>> season(timetuple)
    0
    '''
    day_of_year = timetuple.tm_yday
    label = int(((day_of_year +31) % 366)/91.5)

    return label

def month(timetuple):
    '''    
    Extracts the month from dateutil timetuple object, and outputs labels in 0 .. 11

    >>> import dateutil
    >>> timetuple = dateutil.parser.parse('2013-04-07_17-02-21', fuzzy=True).timetuple()
    >>> month(timetuple)
    3

    >>> timetuple = dateutil.parser.parse('2013-01-07_17-12-21', fuzzy=True).timetuple()
    >>> month(timetuple)
    0
    '''
    return timetuple.tm_mon - 1

def week(timetuple):
    '''
    Extracts the week from dateutil timetuple object, and outputs labels in 0 .. 52
    NB weeks start on the day the year started, not calendar weeks

    >>> import dateutil
    >>> timetuple = dateutil.parser.parse('2013-04-07_17-02-21', fuzzy=True).timetuple()
    >>> week(timetuple)
    13

    >>> timetuple = dateutil.parser.parse('2013-01-07_17-12-21', fuzzy=True).timetuple()
    >>> week(timetuple)
    0
    '''
    day_of_year = timetuple.tm_yday
    return (day_of_year - 1) // 7 # day of year integer div 7

def day(timetuple):
    '''
    Extracts the day of the year from dateutil timetuple object, and outputs labels in 0 .. 365

    >>> import dateutil
    >>> timetuple = dateutil.parser.parse('2013-04-07_17-02-21', fuzzy=True).timetuple()
    >>> day(timetuple)
    96

    # first day of the year
    >>> timetuple = dateutil.parser.parse('2013-01-01_17-12-21', fuzzy=True).timetuple()
    >>> day(timetuple)
    0
    '''
    return timetuple.tm_yday - 1

def hour(timetuple, start = 0):
    '''
    Extracts the hour of the day, and returns hour of the day minus the start
    
    >>> timetuple = dateutil.parser.parse('2013-01-01_17-12-21', fuzzy=True).timetuple()
    >>> hour(timetuple, start=12)
    5
    '''
    return timetuple.tm_hour - start

def daytime(timetuple, start, end):
    '''
    Divides the day into four areas, morning, noon, afternoon, evening, by dividing the
    time between the start (hour) and end (hour) into four equally spaced periods
    
    >>> timetuple = dateutil.parser.parse('2013-01-01_17-12-21', fuzzy=True).timetuple()
    >>> daytime(timetuple, start=12, end=24)
    1
    '''
    curr_time = hour(timetuple, start = start) * 60 + timetuple.tm_min
    interval = (end - start) * 60
    end_morning = int(interval * 0.25)
    end_noon = int(interval * 0.5)
    end_afternoon = int(interval * 0.75)

    if curr_time < end_morning:
        return 0
    elif end_morning <= curr_time < end_noon:
        return 1
    elif end_noon <= curr_time < end_afternoon:
        return 2
    else:
        return 3

def gen_labels(list_jpgs, label_type, start=0, end=24):
    '''
    Generates a list of labels of given label_type (season, month, week or day) for a list of timestamped filenames (list_jpgs)

    >>> gen_labels(['/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_14-31-57.jpg',
    ...     '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_14-48-25.jpg',
    ...     '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_15-04-32.jpg',
    ...     '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_15-19-59.jpg',
    ...     '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_15-36-58.jpg']
    ...     ,"month")
    [11, 11, 11, 11, 11]

    >>> gen_labels(['/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_14-31-57.jpg',
    ...         '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_14-48-25.jpg',
    ...         '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_15-04-32.jpg',
    ...         '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_15-19-59.jpg',
    ...         '/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_15-36-58.jpg']
    ...         ,"other")
    Traceback (most recent call last):
    ...
    KeyError: 'other'

    >>> gen_labels(['foo'], 'season')
    Traceback (most recent call last):
      File "/usr/lib/python2.7/doctest.py", line 1289, in __run
        compileflags, 1) in test.globs
      File "<doctest __main__.gen_labels[2]>", line 1, in <module>
        gen_labels(['foo'], 'season')
      File "time_to_label.py", line 143, in gen_labels
        return [label_fun(filename_to_timetuple(filename)) for filename in list_jpgs]
      File "time_to_label.py", line 140, in filename_to_timetuple
        timetuple = dateutil.parser.parse(date_string.replace('_', ' ')).timetuple()
      File "/usr/lib/python2.7/dist-packages/dateutil/parser.py", line 697, in parse
        return DEFAULTPARSER.parse(timestr, **kwargs)
      File "/usr/lib/python2.7/dist-packages/dateutil/parser.py", line 303, in parse
        raise ValueError, "unknown string format"
    ValueError: unknown string format

    >>> 

    >>> gen_labels('/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/2012-12-06_14-31-57.jpg', 'season')
    Traceback (most recent call last):
     ...
    AssertionError

    '''
    assert isinstance(list_jpgs, list)

    times = dict()
    times["season"] = season
    times["month"] = month
    times["week"] = week
    times["day"] = day
    times["hour"] = hour
    times["daytime"] = daytime
    label_fun = times[label_type]

    #pdb.set_trace()

    def filename_to_timetuple(filename):
        date_string = os.path.splitext(os.path.basename(filename))[0]
        timetuple = dateutil.parser.parse(date_string.replace('_', ' ')).timetuple()
        return timetuple

    if label_type in ("season", "month", "week", "day"):
        labels =  [label_fun(filename_to_timetuple(filename)) for filename in list_jpgs]
    elif label_type == "hour":
        labels =  [label_fun(filename_to_timetuple(filename), start=start) for filename in list_jpgs]
    elif label_type == "daytime":
        labels =  [label_fun(filename_to_timetuple(filename), start=start, end=end) for filename in list_jpgs]

    return labels

# def list_jpgs(limit = 10):
#     root = "/scratch_net/biwinator02/voanna/TimePrediction/data/processed/2016-01-07/"
#     return sorted(glob.glob(os.path.join(root, "*jpg")))[:limit]

def num_categories(label_type):
    '''
    Returns number of possible labels for each type of time category


    >>> num_categories('season')
    4

    >>> num_categories('other')
    Traceback (most recent call last):
    ...
    KeyError: 'other'

    '''
    nc = {
        "season" : 4,
        "month" : 12,
        "week" : 53,
        "day" : 366
        }
    return nc[label_type]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
