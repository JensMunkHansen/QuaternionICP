/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "QuaternionICP", "index.html", [
    [ "Dependencies", "index.html#autotoc_md1", null ],
    [ "Building", "index.html#autotoc_md2", [
      [ "1. Build Dependencies", "index.html#autotoc_md3", null ],
      [ "2. Build Project", "index.html#autotoc_md4", null ],
      [ "3. Run Tests", "index.html#autotoc_md5", null ]
    ] ],
    [ "Important: MKL Configuration", "index.html#autotoc_md6", null ],
    [ "Usage", "index.html#autotoc_md7", [
      [ "SingleICP", "index.html#autotoc_md8", null ],
      [ "MultiICP", "index.html#autotoc_md9", null ]
    ] ],
    [ "Project Structure", "index.html#autotoc_md10", null ],
    [ "Documentation", "index.html#autotoc_md11", null ],
    [ "Continuous Integration", "index.html#autotoc_md12", null ],
    [ "Ray-Projection ICP Documentation", "ray_projection_overview.html", [
      [ "Introduction", "ray_projection_overview.html#rpo_intro", null ],
      [ "Mathematical Foundations", "ray_projection_overview.html#rpo_foundations", null ],
      [ "Ray-Projection Jacobians", "ray_projection_overview.html#rpo_jacobians", null ],
      [ "Validation Methodology", "ray_projection_overview.html#rpo_validation", null ],
      [ "Experimental Results", "ray_projection_overview.html#rpo_experimental", null ],
      [ "Suggested Reading Order", "ray_projection_overview.html#rpo_reading", null ],
      [ "Implementation", "ray_projection_overview.html#rpo_implementation", null ]
    ] ],
    [ "SO(3) Left Jacobian", "so3_left_jacobian.html", [
      [ "Naming Convention", "so3_left_jacobian.html#so3lj_naming", null ],
      [ "Context: SE(3) Exponential Map", "so3_left_jacobian.html#so3lj_context", null ],
      [ "Definition", "so3_left_jacobian.html#so3lj_definition", null ],
      [ "Closed-Form Expression", "so3_left_jacobian.html#so3lj_closed_form", null ],
      [ "Special Cases", "so3_left_jacobian.html#so3lj_special_cases", [
        [ "At Zero Rotation", "so3_left_jacobian.html#so3lj_identity", null ],
        [ "Small Angle Approximation", "so3_left_jacobian.html#so3lj_small_angle", null ]
      ] ],
      [ "Derivation", "so3_left_jacobian.html#so3lj_derivation", [
        [ "From the Matrix Exponential", "so3_left_jacobian.html#so3lj_series", null ],
        [ "Using Rodrigues' Formula", "so3_left_jacobian.html#so3lj_rodrigues", null ]
      ] ],
      [ "Numerical Implementation", "so3_left_jacobian.html#so3lj_numerical", null ],
      [ "Inverse (Right Jacobian)", "so3_left_jacobian.html#so3lj_inverse", null ],
      [ "Related Documents", "so3_left_jacobian.html#so3lj_related", null ],
      [ "References", "so3_left_jacobian.html#so3lj_references", null ]
    ] ],
    [ "SE(3) Full Chart Jacobian (Right-Multiplicative)", "se3_full_chart_jacobian.html", [
      [ "Conventions", "se3_full_chart_jacobian.html#se3_full_conventions", null ],
      [ "Definition of the Chart Jacobian", "se3_full_chart_jacobian.html#se3_full_definition", null ],
      [ "Blocks That Are Always Zero", "se3_full_chart_jacobian.html#se3_full_zero_blocks", null ],
      [ "Translation w.r.t. Translation", "se3_full_chart_jacobian.html#se3_full_dt_dv", null ],
      [ "Quaternion w.r.t. Rotation", "se3_full_chart_jacobian.html#se3_full_dq_dw", null ],
      [ "Translation w.r.t. Rotation (Away from delta = 0)", "se3_full_chart_jacobian.html#se3_full_dt_dw", null ],
      [ "Final Forms", "se3_full_chart_jacobian.html#se3_full_final_form", null ],
      [ "Related Documents", "se3_full_chart_jacobian.html#se3_full_related", null ]
    ] ],
    [ "Ray-Projection Residual Jacobians on SE(3)", "ray_projection_jacobian.html", [
      [ "Residual Definition", "ray_projection_jacobian.html#rpj_residual", [
        [ "Forward Case (Source to Target)", "ray_projection_jacobian.html#rpj_forward", null ],
        [ "Backward Case (Target to Source)", "ray_projection_jacobian.html#rpj_backward", null ]
      ] ],
      [ "The Quotient Rule", "ray_projection_jacobian.html#rpj_quotient_rule", null ],
      [ "Translation Jacobian", "ray_projection_jacobian.html#rpj_translation", [
        [ "Numerator Dependence", "ray_projection_jacobian.html#rpj_trans_numerator", null ],
        [ "Denominator Dependence", "ray_projection_jacobian.html#rpj_trans_denominator", null ],
        [ "Translation Jacobian (Both Consistent and Simplified)", "ray_projection_jacobian.html#rpj_trans_jacobian", null ],
        [ "Consequence for Finite Differences", "ray_projection_jacobian.html#rpj_trans_fd", null ]
      ] ],
      [ "Rotation Jacobian", "ray_projection_jacobian.html#rpj_rotation", [
        [ "Both Terms Depend on Rotation", "ray_projection_jacobian.html#rpj_rot_dependence", null ],
        [ "The Missing Term", "ray_projection_jacobian.html#rpj_rot_missing", null ],
        [ "Computing the Denominator Derivative", "ray_projection_jacobian.html#rpj_rot_db_dw", null ]
      ] ],
      [ "When Simplified Fails", "ray_projection_jacobian.html#rpj_when_simplified", null ],
      [ "Validating Without Finite Differences", "ray_projection_jacobian.html#rpj_validation", null ],
      [ "Finite Difference Behavior Summary", "ray_projection_jacobian.html#rpj_fd_behavior", null ],
      [ "Summary", "ray_projection_jacobian.html#rpj_summary", null ]
    ] ],
    [ "Finite-Difference Jacobian Validation on SE(3)", "epsilon_sweep_guide.html", [
      [ "Overview", "epsilon_sweep_guide.html#esg_overview", null ],
      [ "Two Competing Errors", "epsilon_sweep_guide.html#esg_competing_errors", [
        [ "Truncation Error (Large Epsilon)", "epsilon_sweep_guide.html#esg_truncation", null ],
        [ "Roundoff Error (Small Epsilon)", "epsilon_sweep_guide.html#esg_roundoff", null ]
      ] ],
      [ "The Plateau Phenomenon", "epsilon_sweep_guide.html#esg_plateau", [
        [ "Interpreting Results", "epsilon_sweep_guide.html#esg_plateau_interpretation", null ]
      ] ],
      [ "SE(3) Right-Multiplication Convention", "epsilon_sweep_guide.html#esg_se3_convention", null ],
      [ "The Critical Translation Pitfall", "epsilon_sweep_guide.html#esg_critical_pitfall", null ],
      [ "Finite-Difference Formulas", "epsilon_sweep_guide.html#esg_fd_formulas", [
        [ "Translation Components", "epsilon_sweep_guide.html#esg_fd_translation", null ],
        [ "Rotation Components", "epsilon_sweep_guide.html#esg_fd_rotation", null ]
      ] ],
      [ "Why Translation and Rotation Behave Differently", "epsilon_sweep_guide.html#esg_linear_vs_nonlinear", [
        [ "Translation: Linear Dependence", "epsilon_sweep_guide.html#esg_translation_linear", null ],
        [ "Rotation: Nonlinear Dependence", "epsilon_sweep_guide.html#esg_rotation_nonlinear", null ]
      ] ],
      [ "Practical Epsilon Sweep Recipe", "epsilon_sweep_guide.html#esg_practical_recipe", [
        [ "Choose a Logarithmic Sweep", "epsilon_sweep_guide.html#esg_sweep_range", null ],
        [ "Procedure", "epsilon_sweep_guide.html#esg_sweep_procedure", null ],
        [ "Typical Plateau Ranges", "epsilon_sweep_guide.html#esg_typical_ranges", null ]
      ] ],
      [ "Validating Simplified vs Consistent Jacobians", "epsilon_sweep_guide.html#esg_quotient_rule", null ],
      [ "Validation Checklist", "epsilon_sweep_guide.html#esg_checklist", null ],
      [ "Key Takeaways", "epsilon_sweep_guide.html#esg_takeaway", null ]
    ] ],
    [ "Validating Left vs Right Jacobians on SO(3)", "jacobian_validation.html", [
      [ "Problem Setup", "jacobian_validation.html#jv_problem_setup", null ],
      [ "Finite-Difference Validation (Primary Test)", "jacobian_validation.html#jv_fd_validation", [
        [ "Left Jacobian Test", "jacobian_validation.html#jv_left_test", null ],
        [ "Right Jacobian Test", "jacobian_validation.html#jv_right_test", null ],
        [ "Practical Notes", "jacobian_validation.html#jv_practical_notes", null ]
      ] ],
      [ "Cross-Validation Between Left and Right Jacobians", "jacobian_validation.html#jv_cross_validation", null ],
      [ "Simple Sanity Test Function", "jacobian_validation.html#jv_sanity_test", null ],
      [ "Validation Checklist", "jacobian_validation.html#jv_checklist", null ]
    ] ],
    [ "Interpreting FD Results for the Simplified Jacobian", "simplified_jacobian_analysis.html", [
      [ "Measured Results", "simplified_jacobian_analysis.html#sja_results", null ],
      [ "What the Simplified Jacobian is Missing", "simplified_jacobian_analysis.html#sja_missing_term", null ],
      [ "Why the Simplified Error is Around 0.005 to 0.01", "simplified_jacobian_analysis.html#sja_error_magnitude", null ],
      [ "Why the Consistent Jacobian Matches FD", "simplified_jacobian_analysis.html#sja_consistent_validated", null ],
      [ "Why Optimal Epsilon Remains 1e-6 for Simplified", "simplified_jacobian_analysis.html#sja_best_eps", null ],
      [ "What Happens Near Convergence", "simplified_jacobian_analysis.html#sja_convergence", null ],
      [ "Conclusion", "simplified_jacobian_analysis.html#sja_conclusion", null ]
    ] ],
    [ "SE(3) Optimization: 7D Ambient vs 6D Tangent Parameterization", "se3_ambient_vs_tangent.html", [
      [ "Background", "se3_ambient_vs_tangent.html#se3_background", null ],
      [ "Advantages of the 7D Ambient Representation", "se3_ambient_vs_tangent.html#se3_advantages_7d", [
        [ "1. Trigonometry-free state storage", "se3_ambient_vs_tangent.html#autotoc_md15", null ],
        [ "2. Clean separation of state and update", "se3_ambient_vs_tangent.html#autotoc_md17", null ],
        [ "3. Better solver interoperability", "se3_ambient_vs_tangent.html#autotoc_md19", null ],
        [ "4. Numerically stable linearization", "se3_ambient_vs_tangent.html#autotoc_md21", null ],
        [ "5. Quaternion gauge freedom is harmless", "se3_ambient_vs_tangent.html#autotoc_md23", null ],
        [ "6. Easier residual and Jacobian implementation", "se3_ambient_vs_tangent.html#autotoc_md25", null ],
        [ "7. Conceptual alignment with Lie theory", "se3_ambient_vs_tangent.html#autotoc_md27", null ]
      ] ],
      [ "When a Pure 6D Tangent Formulation Makes Sense", "se3_ambient_vs_tangent.html#se3_when_tangent", null ],
      [ "Summary", "se3_ambient_vs_tangent.html#se3_summary", null ]
    ] ],
    [ "Ray-Projection Residual with Incidence Weighting w(c)", "md_Doc_2JacobianPedantic.html", [
      [ "Correct Jacobians (including dw/dc) for SE(3) right-multiplication", "md_Doc_2JacobianPedantic.html#autotoc_md31", null ],
      [ "1) Definitions", "md_Doc_2JacobianPedantic.html#autotoc_md33", null ],
      [ "2) GeometryWeighting: gating, clamping, and modes", "md_Doc_2JacobianPedantic.html#autotoc_md35", null ],
      [ "3) SE(3) right-multiplication perturbation and basic Jacobians", "md_Doc_2JacobianPedantic.html#autotoc_md37", null ],
      [ "4) Full derivative of r = w(c) * a/c", "md_Doc_2JacobianPedantic.html#autotoc_md39", null ],
      [ "5) Computing w(c) and wp = dw/dc for your modes", "md_Doc_2JacobianPedantic.html#autotoc_md41", null ],
      [ "Case A: enable_gate and abs(c) < tau", "md_Doc_2JacobianPedantic.html#autotoc_md42", null ],
      [ "Case B: not enable_weight", "md_Doc_2JacobianPedantic.html#autotoc_md43", null ],
      [ "Case C: enable_weight and not gated", "md_Doc_2JacobianPedantic.html#autotoc_md44", null ],
      [ "6) Backward residual (target -> source) under right-multiplication", "md_Doc_2JacobianPedantic.html#autotoc_md46", null ],
      [ "7) Final Jacobian formula to implement (forward, right-multiplication)", "md_Doc_2JacobianPedantic.html#autotoc_md48", null ],
      [ "8) Notes on nondifferentiabilities", "md_Doc_2JacobianPedantic.html#autotoc_md50", null ],
      [ "9) Summary", "md_Doc_2JacobianPedantic.html#autotoc_md52", null ]
    ] ],
    [ "Effective Conditioning via Manifold Scaling", "effective_conditioning.html", [
      [ "Short Answer", "effective_conditioning.html#cond_short_answer", null ],
      [ "What Ceres Does Internally", "effective_conditioning.html#cond_ceres_internal", null ],
      [ "The Key Rule", "effective_conditioning.html#cond_key_rule", null ],
      [ "Concrete Example: Scaling Rotation vs Translation", "effective_conditioning.html#cond_example", null ],
      [ "Modified Plus Operation", "effective_conditioning.html#cond_modified_plus", null ],
      [ "Modified Chart Jacobian", "effective_conditioning.html#cond_modified_chart", null ],
      [ "What Remains Unchanged", "effective_conditioning.html#cond_unchanged", null ],
      [ "Important Consistency Check", "effective_conditioning.html#cond_consistency", null ],
      [ "Summary", "effective_conditioning.html#cond_summary", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", null ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"BackendFactory_8cpp_source.html",
"md_Doc_2JacobianPedantic.html#autotoc_md39",
"structICP_1_1InnerParams.html#a183d455fe602d1751472ce54b0f0ddc6"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';