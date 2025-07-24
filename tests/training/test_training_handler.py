# tests/agent/test_training_handler.py

# global imports
import logging
from ddt import ddt
from unittest import TestCase
from unittest.mock import Mock, patch

# local imports
from source.agent import AgentHandler, ReinforcementLearningStrategyHandler, PerformanceTestingStrategyHandler
from source.training import TrainingConfig, TrainingHandler

@ddt
class TrainingHandlerTestCase(TestCase):
    """
    Test case for TrainingHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        mock_config = Mock(spec = TrainingConfig)
        mock_config.nr_of_steps = 2000
        mock_config.nr_of_episodes = 10
        mock_config.repeat_test = 1
        mock_config.instantiate_agent_handler.return_value = Mock(spec = AgentHandler)

        self.__sut: TrainingHandler = TrainingHandler(mock_config)
        self.__mocked_agent_handler = mock_config.instantiate_agent_handler.return_value

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injections of certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                if name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    def test_training_handler_run_training(self) -> None:
        """
        Tests TrainingHandler's run_training functionality.

        Verifies that the run_training method correctly delegates to the agent handler
        for training and testing. It checks that the agent handler's train_agent and
        test_agent methods are called with the correct parameters.

        Asserts:
            The AgentHandler's train_agent and test_agent methods are called once.
        """

        logging.info("Attempting to instantiate environment from TrainingConfig.")
        mocked_train_key_1 = "mocked_train_key_1"
        mocked_train_key_2 = "mocked_train_key_2"
        mocked_train_data_1 = {"mocked_train_metric_1": 1}
        mocked_train_data_2 = {"mocked_train_metric_2": 2}
        mocked_test_key_1 = "mocked_test_key_1"
        mocked_test_key_2 = "mocked_test_key_2"
        mocked_test_data_1 = {"mocked_test_metric_1": 1}
        mocked_test_data_2 = {"mocked_test_metric_2": 2}
        self.__mocked_agent_handler.print_model_summary.return_value = "Mocked model summary"
        self.__mocked_agent_handler.train_agent.return_value = ([mocked_train_key_1, mocked_train_key_2],
                                                                [mocked_train_data_1, mocked_train_data_2])
        self.__mocked_agent_handler.test_agent.return_value = ({0: [mocked_test_key_1, mocked_test_key_2]},
                                                               {0: [mocked_test_data_1, mocked_test_data_2]})

        logging.info("Invoking run_training.")
        self.__sut.run_training()

        logging.info("Validating expected calls.")
        self.__mocked_agent_handler.print_model_summary.assert_called_once()
        self.__mocked_agent_handler.train_agent.assert_called_once()
        self.__mocked_agent_handler.test_agent.assert_called_once()
        self.assertEqual(self.__sut._TrainingHandler__generated_data, {
            'train': {
                mocked_train_key_1: mocked_train_data_1,
                mocked_train_key_2: mocked_train_data_2
            },
            'test': {
                0: {
                    mocked_test_key_1: mocked_test_data_1,
                    mocked_test_key_2: mocked_test_data_2
                }
            }
        })

    @patch('reportlab.pdfgen.canvas.Canvas', new_callable = Mock)
    def test_training_handler_generate_report(self, mock_pdf: Mock = Mock()) -> None:
        """
        Tests TrainingHandler's generate_report functionality.

        Verifies that the generate_report method correctly creates a PDF report
        containing training and testing results. It mocks the PDF generation process
        and checks that the appropriate methods are called on the PDF canvas object.

        Parameters:
            mock_pdf (Mock): Mock for the reportlab Canvas class to track PDF operations.

        Asserts:
            The PDF's title is set correctly.
            Text content is added to the PDF.
            Images (plots) are added to the PDF.
            Text labels are added to the PDF.
            The PDF is saved after creation.
        """

        logging.info("Attempting to generate report.")
        self.__update_sut(_TrainingHandler__generated_data = {
            'train':
                {
                    ReinforcementLearningStrategyHandler.PLOTTING_KEY: {
                        'nb_steps': [100, 200],
                        'episode_reward': [0.9, 0.3]
                    }
                 },
            'test': {
                0: {
                        PerformanceTestingStrategyHandler.PLOTTING_KEY: {
                            'assets_values': [1000, 1100],
                            'currency_prices': [80000, 81000],
                            'iterations': [10, 20],
                            'solvency_coefficient': 10
                        }
                    }
                }
            }
        )

        mock_pdf.return_value = Mock()
        mock_pdf.return_value.stringWidth.return_value = 100
        mocked_report_path = "mocked/report/path"
        self.__sut.generate_report(mocked_report_path)

        logging.info("Validating expected calls.")
        mock_pdf.return_value.setTitle.assert_called_once_with("Report")
        mock_pdf.return_value.drawText.assert_called()
        mock_pdf.return_value.drawImage.assert_called()
        mock_pdf.return_value.drawString.assert_called()
        mock_pdf.return_value.save.assert_called_once()
