# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .python import ensure_installed
ensure_installed("prefect")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from .. import graphql,string
from time import sleep
import prefect
client=None

def connect(prefect_key:str=None)->None:
    global client
    if prefect_key is None:
        string.read_config_file(
            file="settings.ini",
            section="PREFECT",
            variables="prefect_key",
            object=locals(),
        )
    if prefect_key:
        client = prefect.Client(api_key=prefect_key )
        graphql.connect(client)
    else:
        client=None
    print(f"prefect_key={prefect_key}")

def get_schema() -> dict:
    if client: return graphql.query_schema()

def get_flows_and_runs(flow_fields:str="id,name",run_fields:str="id,state,labels,start_time",status:str=None) -> dict:
    variables={}
    if status: variables["status"]=status
    if client:
        query="""

                        query ($status: String) {
                            flow {
                                FLOW_FIELDS,
                                flow_runs(
                                    order_by: {start_time: desc}, 
                                    where: {state: {_eq: $status}}
                                ) {RUN_FIELDS}
                            }
                        }

                """
        query=query.replace("FLOW_FIELDS",flow_fields).replace("RUN_FIELDS",run_fields)
        flows=graphql.execute(query=query, variables=variables).get("data",{}).get("flow",[])
        if status:
            flows=[flow for flow in flows if len(flow.get("flow_runs",[]))>0]
        return flows

def get_running_flows(flow_id:str=None,except_flow_id:str=None,except_flowrun_id:str=None,allof_labels:set=(),anyof_labels:set=())->dict:
    """
        flow_id - can be used to check if an instance of the same flow is already running.
        ie, no need to do inference if previous one is still running

        except_flow_id - can be used to check if instance if OTHER flows with some labels are already running.
        ie, allows training flow to wait till other ML tasks have completed

    """
    flows=get_flows_and_runs(status="Running")
    if flows:
        results=[]
        for flow in flows:
            cur_flow_id=flow.get("id")
            if flow_id:
                if cur_flow_id!=flow_id: continue
            if except_flow_id:
                if cur_flow_id==except_flow_id: continue

            for flow_run in flow.get("flow_runs",[]):
                flow_run_id=flow_run.get("id")
                if except_flowrun_id:
                    if except_flowrun_id==flow_run_id: continue
                flow_labels=set(flow_run.get("labels",[]))
                if len(anyof_labels)>0:
                    if anyof_labels.intersection(flow_labels):
                        results.append(flow)
                        break
                elif len(allof_labels)>0:
                    if allof_labels.issubset(flow_labels):
                        results.append(flow)
                        break
                else:
                    results.append(flow)
        return results

def wait_for_absense_of_tasks(
    flow_id:str=None,
    except_flowrun_id:str=None,
    labels: set = set(),
    sleep_seconds: int = 10,
    max_retries: int = 60,
    logger: object = None,
):
    n = 0
    while True:
        if get_running_flows(flow_id=flow_id, except_flowrun_id=except_flowrun_id,allof_labels=set(["ml", "gpu"])):
            n += 1
            if n > max_retries:
                return False
            if logger:
                logger.warning(
                    f"Sleeping for {sleep_seconds} seconds ({n}/{max_retries})..."
                )
            sleep(sleep_seconds)
        else:
            return True