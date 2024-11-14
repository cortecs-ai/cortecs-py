import os
import time
from typing import Optional, Dict, Any

import requests
from tqdm import tqdm
import logging


class Cortecs:
    def __init__(self, client_id: str = None, client_secret: str = None,
                 api_base_url: str = 'https://cortecs.ai/api/v1'):
        self.__client_id = client_id if client_id else os.environ.get('CORTECS_CLIENT_ID')
        self.__client_secret = client_secret if client_secret else os.environ.get('CORTECS_CLIENT_SECRET')
        if not self.__client_id or not self.__client_secret:
            raise ValueError("Set `CORTECS_CLIENT_ID` and `CORTECS_CLIENT_SECRET` as environment variable.")

        self.api_base_url = os.environ.get('CORTECS_API_BASE_URL', api_base_url)
        self.token = None
        self.token_expiry = 0

    def _get_token(self) -> None:
        """Private method to get a new token using client credentials."""
        response = requests.post(
            self.api_base_url + '/oauth2/token',
            json={
                'grant_type': 'client_credentials',
                'client_id': self.__client_id,
                'client_secret': self.__client_secret,
            }
        )

        response.raise_for_status()

        token_data = response.json()
        self.token = token_data['access_token']
        self.token_expiry = time.time() + token_data['expires_in']

    def _ensure_token(self) -> None:
        """Private method to ensure the token is valid."""
        if self.token is None or time.time() >= self.token_expiry:
            self._get_token()

    def _request(self, method: str, endpoint: str, auth_required: bool = True, **kwargs) -> requests.Response:
        """Private method to handle API requests with optional token management."""
        if auth_required:
            self._ensure_token()

        headers = {}
        if auth_required and self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        response = requests.request(
            method,
            f'{self.api_base_url}{endpoint}',
            headers=headers,
            **kwargs
        )

        response.raise_for_status()
        return response

    def _get(self, endpoint: str, auth_required: bool = True, **kwargs) -> Dict[str, Any]:
        response = self._request('GET', endpoint, auth_required=auth_required, **kwargs)
        return response.json()

    def _post(self, endpoint: str, data: Optional[Dict] = None, auth_required: bool = True, **kwargs) -> Dict[str, Any]:
        response = self._request('POST', endpoint, auth_required=auth_required, json=data, **kwargs)
        return response.json()

    def restart_instance(self, instance_id: str) -> Dict[str, Any]:
        return self._post('/models/restartModel',
                          data={'id': instance_id},
                          auth_required=True)
    
    def _get_defaults_for_model(self, model_name: str): #TODO get instance_type specific default cotext_length
        response = self._get(endpoint="/models/model", auth_required=False, params={"modelName": model_name})
        default_config = response["hardware_info"]["recommended_config"]
        default_context_length = response["hardware_info"]["hardware_configs"][default_config]["params"]["max_context_length"]
        
        return (default_config, default_context_length)
    
    def _check_equality(self, model_name: str, 
                        existing_instance_type: str, 
                        existing_context_length: int, 
                        req_instance_type: str = None, 
                        req_context_length: int = None):
        if not req_instance_type or not req_context_length:
            default_config, default_context_length = self._get_defaults_for_model(model_name)
            if not req_instance_type:
                req_instance_type = default_config
            if not req_context_length:
                req_context_length = min(default_context_length, 32000)
        
        if req_instance_type == existing_instance_type and req_context_length == existing_context_length:
            return True
        else:
            return False
        
    def start(self, model_name: str, instance_type: str = None, context_length: int = None, force: bool = False) -> Dict[str, Any]: #TODO if context length passed and instance_type not, get minimum
        model_name = model_name.replace('/', '--')  # transform huggingface format

        # check if equivalent model is already in console
        instance_id = None
        if not force:
            res = self.get_instances_status()
            for instance in res['instanceStatus']:
                if instance['model_id'] == model_name:
                    if self._check_equality(model_name, instance["instance_name"], instance["context_length"], instance_type, context_length):
                        instance_id = instance['id']
                        if instance['model_status'] == 'stopped':
                            logging.info(f'instance {instance["model_id"]} is restarted.')
                            self.restart_instance(instance['id'])
                        logging.info(
                            f'{model_name} is already running. No new instance is started (otherwise set force=True).')
                        break

        if not instance_id:
            instance = self._post('/models/startModel',
                                  data={
                                      'model_name': model_name,
                                      'instance_name': instance_type,
                                      'context_length': context_length
                                  },
                                  auth_required=True)
            instance_id = instance['id']
            # todo instance_status is not propagated immediately
            time.sleep(3)

        # todo /models/startModel should return instanceStatus right away
        instance_status = self.get_instance_status(instance_id)
        return instance_status

    def stop(self, model_id: str) -> Dict[str, Any]:
        return self._post('/models/stopModel',
                          data={'id': model_id},
                          auth_required=True)

    def get_instances_status(self) -> Dict[str, Any]:
        return self._get('/models/myModels')

    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        res = self.get_instances_status()
        for instance in res['instanceStatus']:
            if instance['id'] == instance_id:
                return instance

    # hide irrelevant information from user
    def _filter_instance_status(self, instance_status: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
        return instance_status['id'], {
            'base_url': f'https://{instance_status["domain"]}/v1',
            'model_name': instance_status['model_id'].replace('--', '/')
        }

    def start_and_poll(self, model_name: str, instance_type: str = None, context_length: int = None,
                       force: bool = False, poll_interval: int = 5,
                       max_retries: int = 150) -> tuple[str, Dict[str, str]]:
        instance_status = self.start(model_name, instance_type, context_length, force=force)

        n_required_steps = instance_status['init_status']['num_steps'] + 1
        if not force and instance_status['init_status']['status'] == n_required_steps:
            return self._filter_instance_status(instance_status)

        with tqdm(total=n_required_steps, desc='start instance') as pbar:
            for attempt in range(max_retries - 1):
                instance_status = self.get_instance_status(instance_status['id'])
                if instance_status['model_status'] == 'stopped':
                    raise RuntimeError("Instance has been stopped.")
                pbar.set_description(instance_status['init_status']['description'])
                pbar.update(instance_status['init_status']['status'] - pbar.n)

                if instance_status['init_status']['status'] == n_required_steps:  # instance is ready
                    return self._filter_instance_status(instance_status)

                time.sleep(poll_interval)

        raise TimeoutError(f"Timeout after {round(max_retries * poll_interval, 1)} seconds.")