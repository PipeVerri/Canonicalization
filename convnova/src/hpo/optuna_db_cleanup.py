import sqlite3
from pathlib import Path
from pprint import pprint

def remove_two_failed_trials_in_a_row(study_name: str, optuna_db_path: Path):
    conn = sqlite3.connect(optuna_db_path)
    try:
        # Get the study id
        cursor = conn.execute(f"SELECT * from studies where study_name = '{study_name}'")
        study = cursor.fetchone()
        if study is None:
            # Check if it's because of a new study
            cursor = conn.execute(f"SELECT * from studies")
            studies = cursor.fetchall()
            print("Studies are:")
            pprint(studies)
            print(f"DB path is {optuna_db_path}")
            if len(studies) == 0:
                print(f"Study {study_name} not found, but no studies exist in the database. This is likely a new study.")
                return

            raise RuntimeError(f"Study {study_name} not found")
        else:
            study_id = study[0]

        # Check if there are two failed trials in a row in that study
        cursor = conn.execute(f"SELECT * FROM trials WHERE study_id = {study_id} ORDER BY trial_id DESC LIMIT 2")
        trials = cursor.fetchall()
        pprint(f"Trials: {trials}")
        if trials is not None and len(trials) >= 2:
            tables_to_delete = [
                "trials",
                "trial_user_attributes",
                "trial_system_attributes",
                "trial_params",
                "trial_values",
                "trial_intermediate_values",
                "trial_heartbeats",
            ]

            if trials[0][3] == "FAIL" and trials[1][3] == "FAIL":
                for t in tables_to_delete:
                    conn.execute(f"DELETE FROM '{t}' WHERE trial_id = {trials[0][0]};")

        conn.commit()
                        
    except Exception as e:
        conn.close()
        raise e
    conn.close()