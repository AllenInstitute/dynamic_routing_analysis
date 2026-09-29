import json
import pathlib
import threading
import time

import aind_session
import codeocean.capsule
import codeocean.computation
import codeocean.data_asset
import polars as pl

client = aind_session.get_codeocean_client()

print(client)
result_prefix = 'v289'
run_id = "2"
SHORT_TIME_SLEEP = 3
LONG_TIME_SLEEP = 60
N_CONCURRENT_COMPUTATIONS = 20


def run_encoding(session_id: str):
    run_params = codeocean.computation.RunParams(
        capsule_id="63b499b3-cfde-4224-8c38-c5fb3d541e34",  # write cache
        named_parameters=[
            codeocean.computation.NamedRunParam(
                param_name="single_session_id_to_use",
                value=session_id,
            ),
            codeocean.computation.NamedRunParam(
                param_name="result_prefix",
                value=str(result_prefix),  # required
            ),
            codeocean.computation.NamedRunParam(
                param_name="run_id",
                value=str(run_id),
            ),
            codeocean.computation.NamedRunParam(
                param_name="test",
                value="0",  # all values must be supplied as strings
            ),
            codeocean.computation.NamedRunParam(
                param_name="use_process_pool",
                value="False",  # all values must be supplied as strings
            ),
            codeocean.computation.NamedRunParam(
                param_name="override_params_json",
                value='{"time_of_interest": "quiescent", "spike_bin_width": 0.1, "skip_existing": 0, "run_linear_shift": 1, "run_dropout": 1}', # all values must be supplied as strings
            ),
    ],
    )
    computation = client.computations.run_capsule(run_params)
    return computation


url = "https://raw.githubusercontent.com/allenneuraldynamics/dr-datacube/main/assets/datacube_sessions.csv"
session_ids = (
    pl.read_csv(url)
    .filter(pl.col("session_type") != "naive")["session_id"]
    .to_list()
)

# Stop all running jobs:
# for session, id_ in json.loads(pathlib.Path("computations.json").read_text()).items():
#     client.computations.delete_computation(id_)
# exit()


session_ids_16gb = ["713655_2024-08-07", "706401_2024-04-22"]

sessions_to_run = set(session_ids)
sessions_running: set[str] = set()
sessions_succeeded = set()
session_id_to_computation: dict[str, codeocean.computation.Computation] = dict()

def save_all() -> None:
    pathlib.Path("glm_queue.json").write_text(
        json.dumps(
            {
            "session_id_to_computation": {
                    session_id: session_id_to_computation[session_id].id
                    for session_id in session_id_to_computation
                },
            "sessions_to_run": list(sessions_to_run),
            "sessions_running": list(sessions_running),
            "sessions_succeeded": list(sessions_succeeded),
            },
            indent=4,
        )
    )

def check_running_sessions():
    while sessions_to_run or sessions_running:
        for session_id in list(sessions_running):
            time.sleep(SHORT_TIME_SLEEP)  # Sleep to avoid hitting the API rate limit
            computation = client.computations.get_computation(session_id_to_computation[session_id].id)
            if computation.state not in (codeocean.computation.ComputationState.Failed, codeocean.computation.ComputationState.Completed):
                continue
            if computation.end_status == codeocean.computation.ComputationEndStatus.Succeeded:
                sessions_running.remove(session_id)
                sessions_succeeded.add(session_id)
                save_all()
                print(f"Computation {computation.id} for session {session_id} succeeded")
                continue
            if computation.end_status == codeocean.computation.ComputationEndStatus.Failed:
                sessions_running.remove(session_id)
                sessions_to_run.add(session_id)
                save_all()
                print(f"Computation {computation.id} for session {session_id} failed - re-queuing")
                continue
        if len(sessions_running) > 0:
            time.sleep(LONG_TIME_SLEEP)  # Wait a minute before checking all computations again

thread = threading.Thread(target=check_running_sessions, daemon=True)
thread.start()
while len(sessions_succeeded) < len(session_ids):
    if len(sessions_running) >= N_CONCURRENT_COMPUTATIONS:
        time.sleep(LONG_TIME_SLEEP)
        continue
    if not sessions_to_run:
        # no sessions left to run, but there are still some running (which could go back into sessions_to_run)
        time.sleep(LONG_TIME_SLEEP)
        continue
    session_id = sessions_to_run.pop()
    computation = run_encoding(session_id)
    session_id_to_computation[session_id] = computation
    sessions_running.add(session_id)
    save_all()
    print(f"Started computation {computation.id} for session {session_id}")
    time.sleep(SHORT_TIME_SLEEP)  # Sleep to avoid hitting the API rate limit

print("Waiting for all computations to complete (should take no time as all codeocean computations are complete at this point)")
thread.join()
print("All computations completed")

# for session, id_ in json.loads(pathlib.Path("computations.json").read_text()).items():
#     client.computations.delete_computation(id_)

print('writing consolidated results parquet files to S3')
client.computations.run_capsule(
    codeocean.computation.RunParams(
        capsule_id="1003b011-db1f-4c50-b2d3-df7f9dc1dc6e",
        parameters=[str(result_prefix), str(run_id)],
    )
)
