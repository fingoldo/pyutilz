# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed
ensure_installed("dash dash_bootstrap_components flask flask_login dash_html_components")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
from flask import session

from flask_login import current_user

from dash import html
import dash_bootstrap_components as dbc


def get_active_tab_var_name(tabsName: str, prefix: str = "tab"):
    return prefix + "s" + tabsName + "ActiveTab"


def create_tabs(
    tabsName: str,
    tabsList: list,
    draw_tab_content_function,
    content: str = "",
    prefix: str = "tab",
    roles_separator: str = ",",
    tabsClassName: str = None,
    contentClassName: str = None,
    activeLabelClassName: str = None,
    use_cardstyle: bool = False,
):
    """
    Used for easy creation of tabs in Dash, including nested tabs.
    Expected tabs fomat: (label:str, tab_id:str, allowed_user_roles:list or str, tabClassName, labelClassName, tabTooltip)
    """
    # print('In create_tabs of %s' % tabsName)
    user = current_user
    if not user.is_authenticated:
        return

    varName = get_active_tab_var_name(tabsName, prefix=prefix)
    if varName not in session:
        session[varName] = prefix + tabsList[0][0]

    active_tab = session[varName]
    # print('active_tab=%s' % active_tab)

    tabs = []
    tooltips = []
    for tab in tabsList:
        tabClassName, labelClassName, tabTooltip = None, None, None
        tabUsers = None
        tabLabel, tabId, tabUsers, *tabClassNames = tab
        if tabClassNames:
            if len(tabClassNames) > 0:
                tabClassName = tabClassNames[0]
            if len(tabClassNames) > 1:
                labelClassName = tabClassNames[1]
            if len(tabClassNames) > 2:
                tabTooltip = tabClassNames[2]
                # print('tabTooltip=%s' % tabTooltip)
        if tabId is None:
            tabId = tabLabel
        if tabLabel is None:
            tabLabel = tabId
        if type(tabUsers) == str:
            tabUsers = tabUsers.split(roles_separator)
        # if tabTooltip: tooltips.append(dbc.Tooltip(tabTooltip,target=(prefix+tabId)))

        if (tabUsers is None) or (user.role in tabUsers):
            tabs.append(
                dbc.Tab(
                    content,
                    label=tabLabel,
                    tab_id=prefix + tabId,
                    tabClassName=tabClassName,
                    labelClassName=f"{labelClassName} {activeLabelClassName if prefix+tabId==active_tab and activeLabelClassName else ''}",
                )
            )

    if len(tabs) > 0:
        header = dbc.Tabs(tabs, id=prefix + "s" + tabsName, active_tab=active_tab, className=tabsClassName) #, card=use_cardstyle : deprecated
        body = html.Div(draw_tab_content_function(active_tab), id=prefix + "s" + tabsName + "Content")
        if use_cardstyle:
            data = dbc.Card([dbc.CardHeader(header, className="pt-0"), dbc.CardBody(body, className="px-2 py-0")], className="p-0")
        else:
            data = [header, body]
        # print(tooltips)
        # elems.extend(tooltips)
        return dbc.Container(data, fluid=True, className=contentClassName)