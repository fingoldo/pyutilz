"""Deprecated PascalCase/Hungarian-notation compatibility wrappers for ``pyutilz.database.db``.

Each function here is a thin, DeprecationWarning-emitting shim over its snake_case counterpart in
``schema.py``; they are grouped in one module so the modern surface stays free of legacy names and
so the whole compatibility layer can be dropped in a single edit when the deprecation expires.
"""

from typing import Optional

from ._common import warnings

# Delegation goes through the facade (see the PROJECT IDIOM comment in connection.py) so a caller
# or test that replaces e.g. ``pyutilz.database.db.read_table_into_dict`` is honoured by the alias.
import pyutilz.database.db as _facade


def EnsurePgTableExists(sTable: str, sKeyFieldName: Optional[str] = "name", sIdFieldName: Optional[str] = "id", sAutocreateIdTypeName: Optional[str] = None):
    """Deprecated alias for :func:`ensure_pg_table_exists` -- kept for backward compatibility."""
    warnings.warn(
        "EnsurePgTableExists is deprecated and will be removed in a future release; use ensure_pg_table_exists instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _facade.ensure_pg_table_exists(table=sTable, key_field_name=sKeyFieldName, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName)


def ReadTableIntoDic(
    dicEnums: dict,
    sTable: str,
    sKeyFieldName: Optional[str] = "name",
    sCondition: Optional[str] = "",
    sIdFieldName: Optional[str] = "id",
    sAutocreateIdTypeName: Optional[str] = None,
) -> None:
    """Deprecated alias for :func:`read_table_into_dict` -- kept for backward compatibility."""
    warnings.warn(
        "ReadTableIntoDic is deprecated and will be removed in a future release; use read_table_into_dict instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _facade.read_table_into_dict(
        dict_enums=dicEnums, table=sTable, key_field_name=sKeyFieldName, condition=sCondition, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName
    )


def ReadTableIntoDicReversed(
    dicEnums: dict,
    sTable: str,
    sKeyFieldName: Optional[str] = "name",
    sCondition: Optional[str] = "",
    sIdFieldName: Optional[str] = "id",
    sAutocreateIdTypeName: Optional[str] = None,
) -> None:
    """Deprecated alias for :func:`read_table_into_dict_reversed` -- kept for backward compatibility."""
    warnings.warn(
        "ReadTableIntoDicReversed is deprecated and will be removed in a future release; use read_table_into_dict_reversed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _facade.read_table_into_dict_reversed(
        dict_enums=dicEnums, table=sTable, key_field_name=sKeyFieldName, condition=sCondition, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName
    )


def GetIdByKeyFieldAndInsertIfNeeded(
    dicEnums: dict,
    sTable: str,
    sKeyFieldValue: str,
    sKeyFieldName: Optional[str] = "name",
    bKeyIsNotString: Optional[bool] = False,
    sAlternateFieldsNames: Optional[str] = "",
    sAlternateFieldsValues: Optional[str] = "",
    sUniqueConstraintFields: Optional[str] = "",
    bUseAlternateFieldsOnly: Optional[bool] = False,
    sIdFieldName: Optional[str] = "id",
    bAddUpdatedAtTimestamp: Optional[str] = None,
) -> str:
    """Deprecated alias for :func:`get_id_by_key_field_and_insert_if_needed` -- kept for backward compatibility."""
    warnings.warn(
        "GetIdByKeyFieldAndInsertIfNeeded is deprecated and will be removed in a future release; use get_id_by_key_field_and_insert_if_needed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _facade.get_id_by_key_field_and_insert_if_needed(
        dict_enums=dicEnums,
        table=sTable,
        key_field_value=sKeyFieldValue,
        key_field_name=sKeyFieldName,
        key_is_not_string=bKeyIsNotString,
        alternate_fields_names=sAlternateFieldsNames,
        alternate_fields_values=sAlternateFieldsValues,
        unique_constraint_fields=sUniqueConstraintFields,
        use_alternate_fields_only=bUseAlternateFieldsOnly,
        id_field_name=sIdFieldName,
        add_updated_at_timestamp=bAddUpdatedAtTimestamp,
    )
