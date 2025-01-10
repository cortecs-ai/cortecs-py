import os
import time

import requests
from tqdm import tqdm

from cortecs_py.schemas import (
    HardwareType,
    Instance,
    InstanceArgs,
    InstanceStatus,
    ModelPreview,
)


class Cortecs:    
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        api_base_url: str = "https://cortecs.ai/api/v1",
    ) -> None:
        self.__client_id = client_id if client_id else os.environ.get("CORTECS_CLIENT_ID")
        self.__client_secret = client_secret if client_secret else os.environ.get("CORTECS_CLIENT_SECRET")

        self.api_base_url = os.environ.get("CORTECS_API_BASE_URL", api_base_url)
        self.token = None
        self.token_expiry = 0

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        auth_required: bool = True,
        timeout: int = 50,
        **kwargs: dict[str, any],
    ) -> requests.Response:
        """Private method to handle API requests with optional token management."""
        if auth_required:
            self._ensure_token()

        headers = {}
        if auth_required and self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = requests.request(
                method,
                f'{self.api_base_url}{endpoint}',
                headers=headers,
                **kwargs,
                timeout=timeout
            )

            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            raise requests.exceptions.HTTPError(
                f"HTTP error occurred: {http_err}. Response: {response.text}"
            ) from http_err

        return response

    def _get(self, endpoint: str, *, auth_required: bool = True, **kwargs: dict[str, any]) -> dict[str, any]:
        response = self._request("GET", endpoint, auth_required=auth_required, **kwargs)
        return response.json()

    def _post(
        self,
        endpoint: str,
        data: dict[str, any] | None = None,
        *,
        auth_required: bool = True,
        **kwargs: dict[str, any],
    ) -> dict[str, any]:
        response = self._request("POST", endpoint, auth_required=auth_required, json=data, **kwargs)
        return response.json()

    def _delete(self, endpoint: str, *, auth_required: bool = True, **kwargs: dict[str, any]) -> dict[str, any]:
        response = self._request("DELETE", endpoint, auth_required=auth_required, **kwargs)
        return response.json()

    def _get_token(self) -> None:
        """Private method to get a new token using client credentials."""

        response = self._post(
            "/oauth2/token",
            auth_required=False,
            data={
                "grant_type": "client_credentials",
                "client_id": self.__client_id,
                "client_secret": self.__client_secret,
            },
        )

        self.token = response["access_token"]
        self.token_expiry = time.time() + response["expires_in"]

    def _ensure_token(self) -> None:
        """Private method to ensure the token is valid."""
        if self.token is None or time.time() >= self.token_expiry:
            # Lazy check because there are functions that don't require auth
            if not self.__client_id or not self.__client_secret:
                raise ValueError(
                    "Set `CORTECS_CLIENT_ID` and `CORTECS_CLIENT_SECRET` as environment variable "
                    "or pass them to the client constructor."
                )
            self._get_token()

    def _get_default_instance_args(self, instance_args: dict[str, any]) -> InstanceArgs:
        """Private method to get the default instance args for a model."""
        # Future implementation:
        # response = self._get(
        #    "/instances/defaults",
        #    data=data,
        #    auth_required=False,
        # )
        # requested_instance_args = InstanceArgs(**response["default_args"])

        # Temporary implementation:
        model_id = instance_args["model_id"]
        response = self._get(endpoint=f"/models/{model_id}", auth_required=False)["model"]

        hardware_type_defauled = False
        if not instance_args.get("hardware_type_id"):
            instance_args["hardware_type_id"] = response["recommended_config"]
            hardware_type_defauled = True

        if instance_args.get("context_length"):
            max_context_length = response["hardware_configs"][instance_args["hardware_type_id"]][
                "params"
            ]["max_context_length"]
            if instance_args["context_length"] > max_context_length:
                if hardware_type_defauled:
                    raise ValueError(f"Context length {instance_args['context_length']} is too large for the "
                                     f"recommended hardware_type: {instance_args['hardware_type_id']}.")
                else:
                    raise ValueError(
                        f"Context length {instance_args['context_length']} is too large for the selected hardware type:"
                        f" {instance_args['hardware_type_id']}. Max context length: {max_context_length}"
                    )
        else:
            default_context_length = min(
                response["hardware_configs"][instance_args["hardware_type_id"]]["params"][
                    "max_context_length"
                ],
                32000,
            )
            instance_args["context_length"] = default_context_length

        if not instance_args.get("billing_interval"):
            instance_args["billing_interval"] = "per_minute"
        return InstanceArgs(**instance_args)

    def poll_instance(self, instance_id: str, poll_interval: int = 5, max_retries: int = 150) -> Instance:
        """
        Poll an instance by its ID until it is running.
        """
        
        instance = self.get_instance(instance_id)
        if instance.instance_status.status == "stopped":
            raise RuntimeError("Instance is stopped.")
        if instance.instance_status.status == "running":
            return instance

        total_steps = instance.instance_status.init_progress["num_steps"]
        with tqdm(total=total_steps, desc="Provisioning resources") as pbar:
            for _ in range(max_retries):
                instance = self.get_instance(instance_id)
                if instance.instance_status.status == "stopped":
                    raise RuntimeError("Instance has been stopped.")
                pbar.set_description(instance.instance_status.init_progress["description"])
                pbar.update(instance.instance_status.init_progress["current_step"] - pbar.n)

                if instance.instance_status.init_progress["current_step"] == total_steps:
                    return instance

                time.sleep(poll_interval)
        raise TimeoutError("Instance provisioning timed out.")

    def start(
        self,
        model_name: str,
        hardware_type_id: str | None = None,
        context_length: int | None = None,
        billing_interval: str | None = None,
        *,
        poll: bool = True,
        poll_interval: int = 5,
        max_retries: int = 150,
    ) -> Instance:
        """Start a new instance with the specified arguments. If poll is True, wait for the instance to start."""
        response = self._post(
            "/instances/start",
            data={
                "model_id": model_name.replace('/', '--'),
                "hardware_type_id": hardware_type_id,
                "context_length": context_length,
                "billing_interval": billing_interval,
            },
            auth_required=True,
        )
        instance = Instance(**response["instance"])
        if poll:
            instance = self.poll_instance(
                instance.instance_id,
                poll_interval=poll_interval,
                max_retries=max_retries,
            )
        return instance

    def restart(
        self,
        instance_id: str,
        *,
        poll: bool = True,
        poll_interval: int = 5,
        max_retries: int = 150,
    ) -> Instance:
        """
        Restart an instance by its ID. If the instance is already running, raises an error.
        If poll is True, wait for the instance to start.
        """
        response = self._post(
            "/instances/restart",
            data={"instance_id": instance_id},
            auth_required=True,
        )

        instance = Instance(**response["instance"])
        if poll:
            instance = self.poll_instance(
                instance.instance_id,
                poll_interval=poll_interval,
                max_retries=max_retries,
            )
        return instance

    def ensure_instance(
        self,
        model_name: str,
        hardware_type_id: str | None = None,
        context_length: int | None = None,
        billing_interval: str | None = None,
        *,
        poll: bool = True,
        poll_interval: int = 5,
        max_retries: int = 150,
    ) -> Instance:
        """
        Ensure an instance with the specified arguments is running.

        - If an instance with the same arguments is already running, return that instance.
        - If an instance with the same arguments is starting, return that instance.
        - If a stopped instance with the same arguments exists, restart and return that instance.
        - Otherwise, start a new instance with the specified arguments.
        If poll is True, wait for the instance to start.
        """
        instance_args = {
            "model_id": model_name.replace('/', '--'),
            "hardware_type_id": hardware_type_id,
            "context_length": context_length,
            "billing_interval": billing_interval,
        }
        requested_instance_args = self._get_default_instance_args(instance_args)
        all_instances = self.get_all_instances()

        pending_instance_id = None
        stopped_instance_id = None
        for instance in all_instances:
            if instance.instance_args == requested_instance_args:
                if instance.instance_status.status == "running":
                    return instance
                if instance.instance_status.status == "pending":
                    pending_instance_id = instance.instance_id
                if instance.instance_status.status == "stopped":
                    stopped_instance_id = instance.instance_id

        if pending_instance_id:
            if poll:
                return self.poll_instance(pending_instance_id, poll_interval=poll_interval, max_retries=max_retries)
            return self.get_instance(pending_instance_id)
        if stopped_instance_id:
            return self.restart(stopped_instance_id, poll=poll)

        return self.start(
            model_name,
            hardware_type_id,
            context_length,
            billing_interval,
            poll=poll,
            poll_interval=poll_interval,
            max_retries=max_retries,
        )

    def get_instance(self, instance_id: str) -> Instance:
        """Get an instance by its ID."""
        instance = self._get(f"/instances/{instance_id}", auth_required=True)["instance"]
        return Instance(**instance)

    def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get the status of an instance by its ID."""
        return self.get_instance(instance_id).instance_status

    def get_all_instances(self) -> list[Instance]:
        """Get all instances."""
        response = self._get("/instances/", auth_required=True)
        return [Instance(**instance) for instance in response["instances"]]

    def get_running_instances(self) -> list[Instance]:
        """Get all running instances."""
        instances = self.get_all_instances()
        return [instance for instance in instances if instance.instance_status.status == "running"]

    def stop(self, instance_id: str) -> Instance:
        """Stop an instance by its ID."""
        response = self._post(
            "/instances/stop",
            data={"instance_id": instance_id},
            auth_required=True,
        )
        instance = Instance(**response["instance"])
        return instance

    def stop_all(self) -> list[str]:
        """Stop all running instances."""
        res = self._post("/instances/stop-all", auth_required=True)
        return res['instance_ids'] if 'instance_ids' in res else None

    def delete(self, instance_id: str) -> str:
        """Delete an instance by its ID. Only stopped instances can be deleted."""
        return self._delete(f"/instances/{instance_id}", auth_required=True)["instance_id"]

    def delete_all(self, *, force: bool = False) -> list[str]:
        """
        Delete all instances. If force is True, all instances will be deleted regardless of their status.
        Otherwise, only stopped instances will be deleted.
        """
        response = self._delete("/instances/delete-all", auth_required=True, params={"force_deletion": force})[
            "instance_ids"
        ]
        return response

    def get_all_models(self) -> list[ModelPreview]:
        """Get all available models."""
        response = self._get("/models/", auth_required=False)
        return [ModelPreview.from_raw_data(model) for model in response["models"]]

    def get_all_hardware_types(self) -> list[HardwareType]:
        """Get all hardware types."""
        response = self._get("/hardware-types/", auth_required=False)
        return [HardwareType(**hardware_type) for hardware_type in response["hardware_types"]]

    def get_available_hardware_types(self) -> list[HardwareType]:
        """Get all hardware types that are currently available."""
        available_hardware = self._get("/hardware-types/available", auth_required=False)
        return [HardwareType(**hardware_type) for hardware_type in available_hardware["hardware_types"]]
