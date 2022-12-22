class continuing:
    """Checks whether the condition function continues to be true
    for a certain duration.

    Returns True if the `condition` function on Vobj's property with name
    `property_name` is satisfied continuously for time interval greater than
    the given duration.

    Time periods of condition being met will be stored in VObj as a property
    with name of `f"{name}_periods"`, in format of a list of tuples `(start,
    end)`, where `start` and `end` are time relative to the start of video, in
    seconds. This property can be accessed with getv, be used in select_cons,
    etc.

    Attributes:
    ----------
    condition: func(property) -> bool
        Condition function to be checked.
    duration: int
        Duration in seconds.
    name: str
        Name of the attribute to store the time periods.
    property_name: str
        Specified in key of VObjConstraint's filter_cons, name of the property
        to be checked.
    """

    def __init__(self, condition, duration, name):
        self.condition = condition
        self.duration = duration
        # use name given as property name of time periods
        self.name = name

    def __call__(self, obj, property_name):
        cur_frame = obj._ctx.frame_id
        # threshold should be set to the same as tracker's threshold of marking
        # track as lost
        # Currently set to equal ByteTrack's threshold
        threshold = int(obj._ctx.fps / 30.0 * 30)
        if self.condition(obj.getv(property_name)):
            # get the start and end of the potentially continuing period
            period_start = obj.getv(f"{self.name}_start")
            period_end = obj.getv(f"{self.name}_end")
            # if (potential) period is not valid
            # or time interval exceeds threshold, reset period
            if period_start is None or cur_frame - period_end > threshold:
                period_start = cur_frame
                period_end = cur_frame
            else:
                # update period, adding current frame to it
                period_end = cur_frame
            setattr(obj, f"__static_{self.name}_start", period_start)
            setattr(obj, f"__static_{self.name}_end", period_end)
            if period_end - period_start >= self.duration * obj._ctx.fps:
                time_period = (int(period_start / obj._ctx.fps),
                               int(period_end / obj._ctx.fps))
                time_periods = obj.getv(f"{self.name}_periods")
                if time_periods is not None:
                    time_periods = time_periods.copy()
                    if time_period[0] == time_periods[-1][0]:
                        # start of current period is the same as end of last
                        # should be merged
                        time_periods[-1] = time_period
                    else:
                        # else a new period, append
                        time_periods.append(time_period)
                else:
                    time_periods = [time_period]
                setattr(obj, f"__static_{self.name}_periods", time_periods)
                return True
        else:
            # reset potential period if condition is not met
            setattr(obj, f"__static_{self.name}_start", None)
            setattr(obj, f"__static_{self.name}_end", None)
        return False
