import json
import pathlib
import time

import aind_session
import codeocean.capsule
import codeocean.computation
import codeocean.data_asset
import polars as pl
import tqdm

client = aind_session.get_codeocean_client()

result_prefix = "v268"
run_id = "linear_shift_good_blocks"


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
            # codeocean.computation.NamedRunParam(
            #     param_name="override_params_json",
            #     value='{"time_of_interest": "quiescent"}',  # all values must be supplied as strings
            # ),
        ],
    )
    computation = client.computations.run_capsule(run_params)
    return computation


session_ids = (
    pl.scan_parquet(
        "s3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet"
    )
    .filter(
        "is_ephys",
        "is_task",
        "is_annotated",
        "is_production",
        pl.col("issues").list.len() == 0,
    )
    .collect()
)["session_id"]

# Stop all running jobs:
# for session, id_ in json.loads(pathlib.Path("computations.json").read_text()).items():
#     client.computations.delete_computation(id_)
# exit()


session_ids_16gb = ["713655_2024-08-07", "706401_2024-04-22"]

session_to_computation = {}
for session_id in tqdm.tqdm(session_ids, desc="Sessions", unit="session"):
    # if session_id in session_ids_16gb:
    #     print(f"Skipping {session_id} because it is requires more than a capsule with more than 8GB memory")
    #     continue
    print(session_id)
    session_to_computation[session_id] = run_encoding(session_id).id
    pathlib.Path("computations.json").write_text(
        json.dumps(session_to_computation, indent=4)
    )
    time.sleep(5)  # Sleep to avoid hitting the API rate limit

# Wait for all computations to finish:
for session, id_ in json.loads(pathlib.Path("computations.json").read_text()).items():
    computation = client.computations.wait_until_completed(
        client.computations.get_computation(id_),
        polling_interval=60,
        )

if computation.end_status != codeocean.computation.ComputationEndStatus.Succeeded:
    print("at least one computation failed - not writing consolidated results")
    quit()

print("writing consolidated results parquet files to S3")
client.computations.run_capsule(
    codeocean.computation.RunParams(
        capsule_id="1003b011-db1f-4c50-b2d3-df7f9dc1dc6e",
        parameters=[str(result_prefix), str(run_id)],
    )
)
