import xml.etree.ElementTree as ET
from typing import Dict, Any


def _parse_field(field_node: ET.Element) -> Dict[str, Any]:
    """
    Parses a field node from a message.
    """
    field_data = {"name": field_node.get("name")}
    if field_data["name"] is None:
        field_data["name"] = field_node.get("NAME")

    field_data["type"] = field_node.get("type")
    if field_data["type"] is None:
        field_data["type"] = field_node.get("TYPE")

    alt_unit_coef = field_node.get("alt_unit_coef")
    if alt_unit_coef is None:
        alt_unit_coef = field_node.get("ALT_UNIT_COEF")

    if alt_unit_coef is None:
        field_data["alt_unit_coef"] = 1.0
    else:
        field_data["alt_unit_coef"] = float(alt_unit_coef)

    return field_data


def _parse_message(message_node: ET.Element) -> Dict[str, Any]:
    """
    Parses a message node from the XML.
    """
    message_data = {"fields": {}}
    message_name = message_node.get("name")
    if message_name is None:
        message_name = message_node.get("NAME")
    message_data["name"] = message_name

    field_names = []
    for field_node in message_node.findall("field"):
        field = _parse_field(field_node)
        field_names.append(field["name"])
        message_data["fields"][field["name"]] = field
    message_data["field_names"] = field_names
    return message_data


def parse_message_definitions(log_file: str) -> Dict[str, Any]:
    """
    Parses an XML protocol definition from a Paparazzi .log file.
    """
    with open(log_file, "r") as f:
        xml_content = f.read()

    # Find the <protocol> section in the log file
    protocol_start = xml_content.find("<protocol")
    protocol_end = xml_content.find("</protocol>") + len("</protocol>")
    protocol_xml = xml_content[protocol_start:protocol_end]

    if not protocol_xml:
        raise ValueError(f"Could not find <protocol> definition in {log_file}")

    root = ET.fromstring(protocol_xml)

    message_definitions = {}
    for msg_class_node in root.findall("msg_class"):
        for message_node in msg_class_node.findall("message"):
            message = _parse_message(message_node)
            message_definitions[message["name"]] = message

    return message_definitions
