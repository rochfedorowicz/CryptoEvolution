# utils/gradient_handler.py

# global imports
import os
from datetime import datetime
from gradient import NotebooksClient

# local imports
from source.utils import SingletonMeta

class GradientHandler(metaclass = SingletonMeta):
    """
    Responsible for communication and management of Paperspace Gradient services.
    """

    # local constants
    __DEFAULT_START_COMMAND = " & PIP_DISABLE_PIP_VERSION_CHECK=1 jupyter lab --allow-root --ip=0.0.0.0 --no-browser \
        --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True \
        --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True"
    __DEFAULT_CONTAINER_TYPE = 'paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125'

    def __init__(self) -> None:
        """
        Class constructor. Before calling it GRADIENT_API_KEY and GRADIENT_PROJECT_ID
        should be available in as environmental variables.

        Raises:
            RuntimeError: If Paperspace Gradient api key or project ID are not defined.
        """

        GRADIENT_API_KEY = os.getenv('GRADIENT_API_KEY')
        GRADIENT_PROJECT_ID = os.getenv('GRADIENT_PROJECT_ID')

        if not GRADIENT_API_KEY or not GRADIENT_PROJECT_ID:
            raise RuntimeError('Paperspace Gradient api key or project ID not found in environment variables!')

        self.notebooks = NotebooksClient(GRADIENT_API_KEY)
        self.project_id = GRADIENT_PROJECT_ID

    def create_notebook(self, command_to_invoke: str, github_repository_url: str = None,
                        notebook_name: str = datetime.now(), machine_types: list = ['Free-P5000'],
                        timeout: int = 6, environment_dict: dict = dict()) -> str:
        """
        Attempts to create notebook basing on certain github repository, starting command and
        sets needed environmental parameters (e.g. variables, machine type, etc.).

        Parameters:
            command_to_invoke (str): Command to be invoked after notebook is started.
            github_repository_url (str): URL to repository that should be downloaded to notebook.
            notebook_name (str): Name that should be given to notebook.
            machine_types (list): List of demanded machine's types that notebook should be attempted
                to be created on. The first successful creation stops attempts to create notebook for
                machine types that are further in the list. Therefore, machine's types should be ordered
                from the most wanted to the least one.
            timeout (int): Number of hours that notebook should be active for. For free instances maximal
                value for this parameter is 6.
            environment_dict (dict): Dictionary containing defined environmental variables.

        Raises:
            RuntimeError: If approached problem during notebook creation.
        """

        notebook_id = None
        error_to_be_raised = None
        for machine_type in machine_types:
            try:
                notebook_id = self.notebooks.create(machine_type = machine_type,
                                                    container = self.__DEFAULT_CONTAINER_TYPE,
                                                    project_id = self.project_id,
                                                    shutdown_timeout = timeout,
                                                    workspace = github_repository_url,
                                                    command = command_to_invoke + self.__DEFAULT_START_COMMAND,
                                                    environment = environment_dict,
                                                    name = notebook_name)
            except Exception as error:
               error_to_be_raised = error

            if notebook_id is not None:
                break

        if notebook_id is None:
            raise RuntimeError(f"Did not managed to create notebook! Original error: {error_to_be_raised}")

        return notebook_id

    def delete_notebook(self, notebook_id: str) -> None:
        """
        Deletes notebook with the given ID from Paperspace Gradient.

        This method attempts to remove a previously created notebook from
        the Paperspace Gradient platform.

        Parameters:
            notebook_id (str): ID of the notebook to be deleted.

        Raises:
            RuntimeError: If approached problem during notebook deletion.
        """

        try:
            self.notebooks.delete(notebook_id)
        except Exception as error:
            raise RuntimeError(f"Did not managed to delete notebook! Original error: {error}")
