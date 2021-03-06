
import sys
import getopt
import logging
import botocore
import boto3
import time
from packaging import version
from time import sleep
from botocore.exceptions import ClientError

logger = logging.getLogger()
personalize = None

def _get_dataset_group_arn(dataset_group_name):
    dsg_arn = None

    paginator = personalize.get_paginator('list_dataset_groups')
    for paginate_result in paginator.paginate():
        for dataset_group in paginate_result["datasetGroups"]:
            if dataset_group['name'] == dataset_group_name:
                dsg_arn = dataset_group['datasetGroupArn']
                break

        if dsg_arn:
            break

    if not dsg_arn:
        raise NameError(f'Dataset Group "{dataset_group_name}" does not exist; verify region is correct')

    return dsg_arn

def _get_solutions(dataset_group_arn):
    solution_arns = []

    paginator = personalize.get_paginator('list_solutions')
    for paginate_result in paginator.paginate(datasetGroupArn = dataset_group_arn):
        for solution in paginate_result['solutions']:
            solution_arns.append(solution['solutionArn'])

    return solution_arns

def _delete_campaigns(solution_arns):
    campaign_arns = []

    for solution_arn in solution_arns:
        paginator = personalize.get_paginator('list_campaigns')
        for paginate_result in paginator.paginate(solutionArn = solution_arn):
            for campaign in paginate_result['campaigns']:
                if campaign['status'] in ['ACTIVE', 'CREATE FAILED']:
                    logger.info('Deleting campaign: ' + campaign['campaignArn'])

                    personalize.delete_campaign(campaignArn = campaign['campaignArn'])
                elif campaign['status'].startswith('DELETE'):
                    logger.warning('Campaign {} is already being deleted so will wait for delete to complete'.format(campaign['campaignArn']))
                else:
                    raise Exception('Campaign {} has a status of {} so cannot be deleted'.format(campaign['campaignArn'], campaign['status']))

                campaign_arns.append(campaign['campaignArn'])

    max_time = time.time() + 30*60 # 30 mins
    while time.time() < max_time:
        for campaign_arn in campaign_arns:
            try:
                describe_response = personalize.describe_campaign(campaignArn = campaign_arn)
                logger.debug('Campaign {} status is {}'.format(campaign_arn, describe_response['campaign']['status']))
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    campaign_arns.remove(campaign_arn)

        if len(campaign_arns) == 0:
            logger.info('All campaigns have been deleted or none exist for dataset group')
            break
        else:
            logger.info('Waiting for {} campaign(s) to be deleted'.format(len(campaign_arns)))
            time.sleep(20)

    if len(campaign_arns) > 0:
        raise Exception('Timed out waiting for all campaigns to be deleted')

def _delete_solutions(solution_arns):
    for solution_arn in solution_arns:
        try:
            describe_response = personalize.describe_solution(solutionArn = solution_arn)
            solution = describe_response['solution']
            if solution['status'] in ['ACTIVE', 'CREATE FAILED']:
                logger.info('Deleting solution: ' + solution_arn)

                personalize.delete_solution(solutionArn = solution_arn)
            elif solution['status'].startswith('DELETE'):
                logger.warning('Solution {} is already being deleted so will wait for delete to complete'.format(solution_arn))
            else:
                raise Exception('Solution {} has a status of {} so cannot be deleted'.format(solution_arn, solution['status']))
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code != 'ResourceNotFoundException':
                raise e

    max_time = time.time() + 30*60 # 30 mins
    while time.time() < max_time:
        for solution_arn in solution_arns:
            try:
                describe_response = personalize.describe_solution(solutionArn = solution_arn)
                logger.debug('Solution {} status is {}'.format(solution_arn, describe_response['solution']['status']))
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    solution_arns.remove(solution_arn)

        if len(solution_arns) == 0:
            logger.info('All solutions have been deleted or none exist for dataset group')
            break
        else:
            logger.info('Waiting for {} solution(s) to be deleted'.format(len(solution_arns)))
            time.sleep(20)

    if len(solution_arns) > 0:
        raise Exception('Timed out waiting for all solutions to be deleted')

def _delete_event_trackers(dataset_group_arn):
    event_tracker_arns = []

    event_trackers_paginator = personalize.get_paginator('list_event_trackers')
    for event_tracker_page in event_trackers_paginator.paginate(datasetGroupArn = dataset_group_arn):
        for event_tracker in event_tracker_page['eventTrackers']:
            if event_tracker['status'] in [ 'ACTIVE', 'CREATE FAILED' ]:
                logger.info('Deleting event tracker {}'.format(event_tracker['eventTrackerArn']))
                personalize.delete_event_tracker(eventTrackerArn = event_tracker['eventTrackerArn'])
            elif event_tracker['status'].startswith('DELETE'):
                logger.warning('Event tracker {} is already being deleted so will wait for delete to complete'.format(event_tracker['eventTrackerArn']))
            else:
                raise Exception('Solution {} has a status of {} so cannot be deleted'.format(event_tracker['eventTrackerArn'], event_tracker['status']))

            event_tracker_arns.append(event_tracker['eventTrackerArn'])

    max_time = time.time() + 30*60 # 30 mins
    while time.time() < max_time:
        for event_tracker_arn in event_tracker_arns:
            try:
                describe_response = personalize.describe_event_tracker(eventTrackerArn = event_tracker_arn)
                logger.debug('Event tracker {} status is {}'.format(event_tracker_arn, describe_response['eventTracker']['status']))
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    event_tracker_arns.remove(event_tracker_arn)

        if len(event_tracker_arns) == 0:
            logger.info('All event trackers have been deleted or none exist for dataset group')
            break
        else:
            logger.info('Waiting for {} event tracker(s) to be deleted'.format(len(event_tracker_arns)))
            time.sleep(20)

    if len(event_tracker_arns) > 0:
        raise Exception('Timed out waiting for all event trackers to be deleted')

def _delete_filters(dataset_group_arn):
    filter_arns = []

    filters_response = personalize.list_filters(datasetGroupArn = dataset_group_arn, maxResults = 100)
    for filter in filters_response['Filters']:
        logger.info('Deleting filter ' + filter['filterArn'])
        personalize.delete_filter(filterArn = filter['filterArn'])
        filter_arns.append(filter['filterArn'])

    max_time = time.time() + 30*60 # 30 mins
    while time.time() < max_time:
        for filter_arn in filter_arns:
            try:
                describe_response = personalize.describe_filter(filterArn = filter_arn)
                logger.debug('Filter {} status is {}'.format(filter_arn, describe_response['filter']['status']))
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    filter_arns.remove(filter_arn)

        if len(filter_arns) == 0:
            logger.info('All filters have been deleted or none exist for dataset group')
            break
        else:
            logger.info('Waiting for {} filter(s) to be deleted'.format(len(filter_arns)))
            time.sleep(20)

    if len(filter_arns) > 0:
        raise Exception('Timed out waiting for all filter to be deleted')

def _delete_datasets_and_schemas(dataset_group_arn):
    dataset_arns = []
    schema_arns = []

    dataset_paginator = personalize.get_paginator('list_datasets')
    for dataset_page in dataset_paginator.paginate(datasetGroupArn = dataset_group_arn):
        for dataset in dataset_page['datasets']:
            describe_response = personalize.describe_dataset(datasetArn = dataset['datasetArn'])
            schema_arns.append(describe_response['dataset']['schemaArn'])

            if dataset['status'] in ['ACTIVE', 'CREATE FAILED']:
                logger.info('Deleting dataset ' + dataset['datasetArn'])
                personalize.delete_dataset(datasetArn = dataset['datasetArn'])
            elif dataset['status'].startswith('DELETE'):
                logger.warning('Dataset {} is already being deleted so will wait for delete to complete'.format(dataset['datasetArn']))
            else:
                raise Exception('Dataset {} has a status of {} so cannot be deleted'.format(dataset['datasetArn'], dataset['status']))

            dataset_arns.append(dataset['datasetArn'])

    max_time = time.time() + 30*60 # 30 mins
    while time.time() < max_time:
        for dataset_arn in dataset_arns:
            try:
                describe_response = personalize.describe_dataset(datasetArn = dataset_arn)
                logger.debug('Dataset {} status is {}'.format(dataset_arn, describe_response['dataset']['status']))
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    dataset_arns.remove(dataset_arn)

        if len(dataset_arns) == 0:
            logger.info('All datasets have been deleted or none exist for dataset group')
            break
        else:
            logger.info('Waiting for {} dataset(s) to be deleted'.format(len(dataset_arns)))
            time.sleep(20)

    if len(dataset_arns) > 0:
        raise Exception('Timed out waiting for all datasets to be deleted')

    for schema_arn in schema_arns:
        try:
            logger.info('Deleting schema ' + schema_arn)
            personalize.delete_schema(schemaArn = schema_arn)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceInUseException':
                logger.info('Schema {} is still in-use by another dataset (likely in another dataset group)'.format(schema_arn))
            else:
                raise e

    logger.info('All schemas used exclusively by datasets have been deleted or none exist for dataset group')

def _delete_dataset_group(dataset_group_arn):
    logger.info('Deleting dataset group ' + dataset_group_arn)
    personalize.delete_dataset_group(datasetGroupArn = dataset_group_arn)

    max_time = time.time() + 30*60 # 30 mins
    while time.time() < max_time:
        try:
            describe_response = personalize.describe_dataset_group(datasetGroupArn = dataset_group_arn)
            logger.debug('Dataset group {} status is {}'.format(dataset_group_arn, describe_response['datasetGroup']['status']))
            break
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.info('Dataset group {} has been fully deleted'.format(dataset_group_arn))
            else:
                raise e

        logger.info('Waiting for dataset group to be deleted')
        time.sleep(20)

def delete_dataset_groups(dataset_group_arns, region = None):
    global personalize
    personalize = boto3.client(service_name = 'personalize', region_name = region)

    for dataset_group_arn in dataset_group_arns:
        logger.info('Dataset Group ARN: ' + dataset_group_arn)

        solution_arns = _get_solutions(dataset_group_arn)

        # 1. Delete campaigns
        _delete_campaigns(solution_arns)

        # 2. Delete solutions
        _delete_solutions(solution_arns)

        # 3. Delete event trackers
        _delete_event_trackers(dataset_group_arn)

        # 4. Delete filters
        _delete_filters(dataset_group_arn)

        # 5. Delete datasets and their schemas
        _delete_datasets_and_schemas(dataset_group_arn)

        # 6. Delete dataset group
        _delete_dataset_group(dataset_group_arn)

        logger.info(f'Dataset group {dataset_group_arn} fully deleted')
