import logging
import re
import json
from .APIs import call_api_with_auth

RESPONSE_TOOLS = [
    {
        "id": "employee_current_info",
        "allowed_profiles": ["empleado", "admin", "supervisor", "gerencia"],
        "aliases": [
            "Información del empleado autenticado",
            "info_empleado_actual",
            "obtener_info_empleado_actual",
        ],
        "type": "function",
        "name": "obtener_info_empleado_actual",
        "description": "Obtiene informacion del empleado autenticado.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "id": "employees_directory",
        "allowed_profiles": ["admin"],
        "aliases": [
            "Información de todos los empleados",
            "todos_los_empleados",
            "obtener_todos_los_empleados",
        ],
        "type": "function",
        "name": "obtener_todos_los_empleados",
        "description": "Obtiene informacion de todos los empleados. Usar solo si el usuario tiene permiso de administrador.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "id": "payroll_receipt",
        "allowed_profiles": ["empleado", "admin", "supervisor", "gerencia"],
        "aliases": [
            "Recibos de pago por periodo",
            "recibo_pago",
            "obtener_recibo_pago",
        ],
        "type": "function",
        "name": "obtener_recibo_pago",
        "description": "Obtiene el recibo de pago del empleado autenticado para un periodo de nomina.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Periodo de nomina en formato YYYYMM#, donde # es un numero del 1 al 4.",
                }
            },
            "required": ["period"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def normalize_profile(permission_type):
    profile = str(permission_type or "").strip().lower()
    aliases = {
        "administrator": "admin",
        "administrador": "admin",
        "superadmin": "admin",
        "true": "admin",
        "1": "admin",
        "yes": "admin",
        "si": "admin",
        "sí": "admin",
        "employee": "empleado",
        "colaborador": "empleado",
    }
    return aliases.get(profile, profile or "empleado")


def user_has_admin_permission(permission_type):
    return normalize_profile(permission_type) == "admin"


def execute_response_tool_call(
    tool_call, company_id, company_name, employee_number, permission_type
):
    function_name = tool_call.name
    params_str = tool_call.arguments or "{}"
    params = json.loads(params_str) if isinstance(params_str, str) else params_str
    is_admin = user_has_admin_permission(permission_type)

    logging.info(
        "%s|%s|%s| RESPONSE TOOL CALL: %s ARGS: %s",
        employee_number,
        company_id,
        company_name,
        function_name,
        params,
    )

    if function_name == "obtener_todos_los_empleados":
        if not is_admin:
            return "El usuario no tiene acceso a la información."

        response_data = get_plantilla_personal(company_id)
        return response_data or "No hay data regresada de la API."

    if function_name == "obtener_info_empleado_actual":
        response_data = get_info_empleado(company_id, employee_number)
        return (
            response_data
            or "No hay data para este empleado o no hay data regresada de la API."
        )

    if function_name == "obtener_recibo_pago":
        if not isinstance(params, dict) or "period" not in params:
            return "Indicar al usuario que se necesita el periodo (YYYYMM#) para consultar la información."

        period = params["period"]
        pattern = r"^\d{4}(0[1-9]|1[0-2])[1-4]$"
        if not re.match(pattern, period):
            return f"Formato de Periodo Invalido: {period}. Debe ser YYYYMM# donde # es un numbero del 1 al 4."

        response_data = get_payroll_receipt(company_id, employee_number, period)
        return (
            response_data
            or "No hay datos para el periodo o no hay data regresada de la API."
        )

    return f"Function '{function_name}' no definida en el agente."


def get_plantilla_personal(company):
    payload = {
        "company": int(company),
        "employeeType": "1",
        "bankData": "1",
        "personalData": "1",
    }
    response = call_api_with_auth("https://api.grupoono.lat/EeDetail", payload)
    return response


def get_info_empleado(company, employee_number):
    payload = {
        "company": int(company),
        "employeeType": "1",
        "bankData": "1",
        "personalData": "1",
        "EmployeeNumber": employee_number,
    }
    response = call_api_with_auth("https://api.grupoono.lat/EeDetail", payload)
    return response


def get_payroll_receipt(
    company, employee_number, period, pay_group="1000", payroll_type="NN"
):
    payload = {
        "companyId": int(company),
        "employeeNumber": employee_number,
        "payrollPeriod": str(period),
        "payGroup": pay_group,
        "payrollType": payroll_type,
    }
    response = call_api_with_auth("https://api.grupoono.lat/EmployeePaySlip", payload)
    return response
