# Structural Information

## Algorithm Schedule
An algorithm schedule is a list of tuples where every tuple contains the id of the algorithm and the point of time where the timeslot ends:
```
schedule: List[Tuple[int, float]] = [(algorithm_id1, end_of_timeslot1), (algorithm_id2, end_of_timeslot2), (algorithm_id3, -1)]
```
In my case, -1 shows us, that there is no end of the timeslot because the algorithm is the last one chosen in the schedule.
An example of a schedule could be looking like this: 
```
schedule: List[Tuple[int, float]] = [(1, 2), (3, 6), (2, -1)]
```
Important to mention is also that the end_of_timeslot timepoint is included into the runtime of the current algorithm.

Policy for two algorithms on the same timestep:
-> The algorithm with the lower index will be chosen, for example the following two schedule will lead to the same result:
```
[(1, 2), (3, 6), (2, -1)] == [(1, 2), (3, 6), (4, 6), (2, -1)]
```

## Ground truth values
Ground truth values are the actual runtimes that each algorithm need to solve the problem instance. It is a dictionary that maps each algorithm id to the actual runtime:
```
ground_truth_values1: Dict[int, float] = {
    algorithm_id1: runtime1,
    algorithm_id2: runtime2,
    algorithm_id3: runtime3,
    algorithm_id4: runtime4
}
```
Within this dictionary the amount of entries is at least as high as the number of algorithms used in the algorithm schedule.
An example of such a dictionary with ground truth values could be looking like this: 
```
ground_truth_values1: Dict[int, float] = {
    1: 3,
    2: 12,
    3: 3,
    4: 4
}
```