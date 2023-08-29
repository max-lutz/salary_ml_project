import os
from pathlib import Path
import streamlit as st
from typing import Dict
from typing import List
from typing import Text

from pages.src.ui import display_header
from pages.src.ui import display_report
from pages.src.ui import select_period
from pages.src.ui import select_project
from pages.src.ui import select_report
from pages.src.ui import set_page_container_style
from pages.src.utils import EntityNotFoundError
from pages.src.utils import get_reports_mapping
from pages.src.utils import list_periods

PROJECTS_DIR: Path = Path("projects")
REPORTS_DIR_NAME: Text = "reports"


if __name__ == "__main__":

    # Configure some styles
    set_page_container_style()

    # Extract available project names and reports directory name
    projects: List[Text] = []
    for path in os.listdir(PROJECTS_DIR):
        if not path.startswith("."):
            projects.append(path)

    try:

        # Sidebar: Select project (UI) and build reports directory path
        selected_project: Path = PROJECTS_DIR / select_project(projects)
        reports_dir: Path = selected_project / REPORTS_DIR_NAME

        # Sidebar: Select period
        periods: List[Text] = list_periods(reports_dir)
        selected_period: Text = select_period(periods)
        period_dir: Path = reports_dir / selected_period

        # Sidebar: Select report (UI)

        report_mapping: Dict[Text, Path] = get_reports_mapping(period_dir)
        selected_report_name: Text = select_report(report_mapping)
        selected_report: Path = report_mapping[selected_report_name]

        # Display report header (UI)
        display_header(
            project_name=selected_project.name,
            period=selected_period,
            report_name=selected_report_name,
        )

        col_1, col_2 = st.columns((2, 1))
        with col_1:
            with st.expander("How to use this app"):
                st.markdown("""
                    This page allows you to view different reports and tests on data that has been fed tp the model
                    in the past weeks.\n\n You can select the time period and type of report (data quality or data drift)
                    and switch between report and tests. \n\n This page could allow you to detect issues in the data 
                    quality or drift and check if your model is performing correctly.
                    """)
        st.write("")
        # Display selected report(UI)
        display_report(selected_report)

    except EntityNotFoundError as e:
        # If some entity (periods directories, specific period or report files)
        # not found then display error in UI
        st.error(e)

    except Exception as e:
        raise e
