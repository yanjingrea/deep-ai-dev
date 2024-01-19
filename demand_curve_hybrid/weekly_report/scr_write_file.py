from demand_curve_hybrid.weekly_report.scr_export_error import error_codes
from demand_curve_hybrid.weekly_report.scr_export_images import demand_images_codes
from demand_curve_hybrid.weekly_report.scr_get_paths import td

latex_dir = '/Users/wuyanjing/PycharmProjects/presentation/src/'
file_name = f'weekly_report_template_{td.date()}.tex'

with open(f"{latex_dir}{file_name}", "w") as file:

    latex_content = r"""
    %----------------------------------------------------------------------------------------
    %	PACKAGES AND THEMES
    %----------------------------------------------------------------------------------------
    \documentclass[aspectratio=169,xcolor=dvipsnames, t]{beamer}
    \usepackage{fontspec} % Allows using custom font. MUST be before loading the theme!
    \usetheme{SimplePlusREA}
    \usepackage{hyperref}
    \usepackage{graphicx} % Allows including images
    \usepackage{booktabs} % Allows the use of \toprule, \midrule and  \bottomrule in tables
    \usepackage{svg} %allows using svg figures
    \usepackage{tikz}
    \usepackage{makecell}
    \usepackage{wrapfig}
    % ADD YOUR PACKAGES BELOW

    %----------------------------------------------------------------------------------------
    %	TITLE PAGE CONFIGURATION
    %----------------------------------------------------------------------------------------

    \title[short title]{Weekly Report}
    \subtitle{New Launch Condo Demand Curve}

    \author[Surname]{Data Science}
    \institute[Real Estate Analytics]{Real Estate Analytics}
    % Your institution as it will appear on the bottom of every slide, maybe shorthand to save space


    \date{\today} % Date, can be changed to a custom date
    %----------------------------------------------------------------------------------------
    %	PRESENTATION SLIDES
    %----------------------------------------------------------------------------------------
    \begin{document}

        \maketitlepage
        \begin{frame}[t]{Overview} 
            \tableofcontents
        \end{frame}
        
        \makesection{Units Ranking U Curve}
        \begin{frame}{Units Ranking U Curve}
            \begin{columns}
                \begin{column}{0.45\textwidth}
                    \colheader{Abstract}
                    \begin{enumerate}
                        \footnotesize
                        \item \textbf{Goal:} Accurately predict the sales sequence within projects, identifying which unit will sell first.
                        \item \textbf{Visualization Explanation:}
                        Scatter Plot: Each dot illustrates the prediction accuracy for distinct projects.
                        Blue Curve: Demonstrates the accuracy outcomes for a random selection strategy.
                        \item \textbf{Current State:} The majority of prediction dots are positioned above the random selection curve, indicating a 6\% enhanced accuracy over random selection.
                    \end{enumerate}
                \end{column}
    
                \begin{column}{0.45\textwidth}
                    \colheader{Results}
                    {
                        \begin{center}
                            \includegraphics[width=1\textwidth]{images/project_level_u_curve}
                        \end{center}
                    }
                \end{column}
    
    
            \end{columns}
        \end{frame}
        
        \makesection{Demand Curve}
        
        \subsection{Introduction}
        
        \begin{frame}{Methodology}
            \begin{columns}
                \footnotesize
                \begin{column}{0.45\textwidth}
                    \colheader{What's Demand Curve?}
                    \begin{enumerate}
                        \item A demand curve is a graphical representation that illustrates the relationship between the price and the quantity demanded by consumers at that price.
                        \item Typically, it slopes downwards from left to right, indicating that as the price of a product decreases, the demand for it generally increases, and vice versa.
                    \end{enumerate}
                \end{column}
    
                \begin{column}{0.45\textwidth}
                    \colheader{What's the Factors Incorporated in Our Model?}
                    {
                        \begin{enumerate}
                            \item \textbf{Project’s Features:} Project size, unit mix, location and so on
                            \item \textbf{Comparable Projects:} Performance and characteristics of nearby and comparable condos
                            \item \textbf{Neighbourhood/Region Characteristics:} market dynamics
                            \item \textbf{National Housing Industry Trends:} housing price index and new condo supply
                        \end{enumerate}
                    }
                \end{column}
    
    
            \end{columns}
        \end{frame}
        
        \subsection{Demand Prediction Error}
        %----------------------------------------------------------------------------------------
        % Prediction Error
        %----------------------------------------------------------------------------------------
        """ + error_codes + r"""

        \subsection{Demand Curve Image}
        %----------------------------------------------------------------------------------------
        % Demand Curve
        %----------------------------------------------------------------------------------------
        """ + demand_images_codes + r""" 

        \makesection{Disclaimer}
        %----------------------------------------------------------------------------------------
        % Disclaimer Page
        %----------------------------------------------------------------------------------------
        \begin{frame}[fragile]
            \frametitle{Disclaimer}
            \footnotesize
            This report and all information and documents in any oral, 
            hardcopy, or electronic form prepared specifically for it (collectively, 'Information') have been 
            prepared by, or on behalf of, Real Estate Analytics. It is provided in summary form and Real Estate Analytics does not warrant or represent 
            that the Information is accurate, current, or complete. The Information is general information only and 
            is not, nor is intended to be, professional or legal advice to a user. Users requiring information beyond 
            that of a general nature in relation to, in connection with, or referred to in the Information, 
            should seek independent professional advice relevant to their own particular circumstances. The 
            Information may include views or recommendations of third parties and does not necessarily reflect the 
            views of Real Estate Analytics or indicate a commitment to a particular course of action. Real Estate 
            Analytics is not liable or responsible to any person for any injury, loss, or damage of any nature 
            whatsoever arising from or incurred by the use of, reliance on, or interpretation of the Information. Any 
            unauthorized use of the Information is strictly prohibited. Users are not authorized to copy, circulate, 
            disclose, disseminate, or distribute the Information, in whole or in part, to any third party, 
            unless explicitly agreed upon by Real Estate Analytics.
        \end{frame}

        %----------------------------------------------------------------------------------------
        % Final PAGE
        %----------------------------------------------------------------------------------------
        \finalpagetext{Thank you for your attention}
        \makefinalpage
    \end{document}
    """

    print(latex_content)

    file.write(
        latex_content
    )