# utils/dynamic_from_string_converter.py

# global imports
import importlib
import logging
import pkgutil
from anytree import Node
from typing import Optional
from types import ModuleType

# local imports
from source.utils import SingletonMeta

class PackageNode(Node):
    """
    Implements a node in the package tree structure. Each node represents a package
    and contains its name, the package itself, and a reference to its parent node.
    """

    def __init__(self, name: str, package: ModuleType, parent: Optional['PackageNode'] = None) -> None:
        """
        Class constructor. Initializes the PackageNode with a name, package, and optional parent node.

        Parameters:
            name (str): The name of the package.
            package (ModuleType): The package module.
            parent (Optional[PackageNode]): The parent node in the tree structure.
        """

        super().__init__(name, parent)
        self.package: ModuleType = package

class DynamicFromStringConverter(metaclass = SingletonMeta):
    """
    Implements a dynamic class converter that allows for the conversion of class names
    from strings to actual class handles. It builds a tree structure of packages and allows
    for the retrieval of class handles based on their names.
    """

    def __init__(self) -> None:
        """
        Class constructor. Initializes an empty list to hold registered packages.
        """

        self.__registered_packages: list[PackageNode] = []

    def __build_package_tree(self, package_name: str, parent: Optional[PackageNode] = None) -> PackageNode:
        """
        Builds a tree structure of packages starting from the given package name.

        Parameters:
            package_name (str): The name of the package to build the tree for.
            parent (Optional[PackageNode]): The parent node in the tree structure.

        Returns:
            (PackageNode): The root node of the package tree.
        """

        root_node: Optional[PackageNode] = None
        try:
            package = importlib.import_module(package_name)
            root_node = PackageNode(name = package_name, package = package, parent = parent)

            if hasattr(package, '__path__'):
                for _, module_name, _ in pkgutil.iter_modules(package.__path__, package_name + '.'):
                    if '.test' not in module_name:
                        self.__build_package_tree(module_name, parent = root_node)
        except:
            logging.warning(f"Failed to import {package_name}. Skipping...")

        return root_node

    def __find_class_in_package_tree(self, class_name: str, package_node: PackageNode) -> Optional[type]:
        """
        Searches for a class in the package tree.

        Parameters:
            class_name (str): The name of the class to search for.
            package_node (PackageNode): The root node of the package tree to search.

        Returns:
            (Optional[type]): The class type if found, None otherwise.
        """

        if hasattr(package_node.package, class_name):
            return getattr(package_node.package, class_name)

        for child in package_node.children:
            found_class = self.__find_class_in_package_tree(class_name, child)
            if found_class is not None:
                return found_class

        return None


    def register_packages(self, packages_to_get_registered: list[str]) -> None:
        """
        Registers a list of packages by building their package trees.

        Parameters:
            packages_to_get_registered (list[str]): A list of package names to register.
        """

        self.__registered_packages: list[PackageNode] = [self.__build_package_tree(package_name) \
                                                         for package_name in packages_to_get_registered]

    def get_class_handle(self, class_name: str) -> type:
        """
        Retrieves the class handle for a given class name from the registered packages.

        Parameters:
            class_name (str): The name of the class to retrieve.

        Raises:
            ValueError: If the class is not found in the registered packages.

        Returns:
            (type): The class type if found.
        """

        for package_tree in self.__registered_packages:
            found_class = self.__find_class_in_package_tree(class_name, package_tree)
            if found_class is not None:
                return found_class

        raise ValueError(f"Class '{class_name}' not found in registered packages.")
