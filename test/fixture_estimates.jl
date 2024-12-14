include("../src/struct.jl")

expected_estimates = [
    FixedEffect(
      id = "(Intercept)",
      estimate = 11.3844061610093,
      std_error = 0.658334898493977,
      lb_ci_alpha05 = 10.0008078566275,
      ub_ci_alpha05 = 12.7680044653911,
      num_df = 1,
      den_df = 17.9118604743733,
      t_statistic = 17.2927277394113,
      p_value = 1.27345593752807E-12
    ),
    FixedEffect(
      id = "FR",
      estimate = 1.01038901897171,
      std_error = 0.206239955266875,
      lb_ci_alpha05 = 0.563034966684026,
      ub_ci_alpha05 = 1.45774307125939,
      num_df = 1.0,
      den_df = 12.5049432612851,
      t_statistic = 4.89909444396584,
      p_value = 0.000325148140252986
      ),
    FixedEffect(
      id = "MC",
      estimate = -1.47173815414142,
      std_error = 0.204216036071839,
      lb_ci_alpha05 = -1.91483948513879,
      ub_ci_alpha05 = -1.02863682314404,
      num_df = 1.0,
      den_df = 12.468492529527,
      t_statistic = -7.20677074362413,
      p_value = 8.69705055437346E-06
    ),
    FixedEffect(
      id = "SS",
      estimate = 0.726682020385344,
      std_error = 0.206239955266875,
      lb_ci_alpha05 = 0.279327968097662,
      ub_ci_alpha05 = 1.17403607267303,
      num_df = 1,
      den_df = 12.5049432612851,
      t_statistic = 3.52347836501912,
      p_value = 0.00395435727812429
    ),
    FixedEffect(
      id = "FR & MC",
      estimate = 0.423845239445128,
      std_error = 0.22394199480273,
      lb_ci_alpha05 = -0.062784810935777,
      ub_ci_alpha05 = 0.910475289826033,
      num_df = 1.0,
      den_df = 12.2955552048801,
      t_statistic = 1.8926563542425,
      p_value = 0.0821789349827481
    ),
    FixedEffect(
      id = "FR & SS",
      estimate = -0.0701331433813935,
      std_error = 0.221201864137814,
      lb_ci_alpha05 = -0.551010786998506,
      ub_ci_alpha05 = 0.410744500235719,
      num_df = 1,
      den_df = 12.2479293440409,
      t_statistic = -0.317054938278905,
      p_value = 0.75654133734547
    ),
    FixedEffect(
      id = "MC & SS",
      estimate = 0.211154760554872,
      std_error = 0.22394199480273,
      lb_ci_alpha05 = -0.275475289826033,
      ub_ci_alpha05 = 0.697784810935777,
      num_df = 1.0,
      den_df = 12.2955552048801,
      t_statistic = 0.942899346506569,
      p_value = 0.363891230681868
    ),
    FixedEffect(
      id = "FR & FR",
      estimate = 0.327581364951284,
      std_error = 0.438634194636724,
      lb_ci_alpha05 = -0.625423625014984,
      ub_ci_alpha05 = 1.28058635491755,
      num_df = 1,
      den_df = 12.3141152725095,
      t_statistic = 0.746821312512096,
      p_value = 0.469191333305121
    ),
    FixedEffect(
      id = "MC & MC",
      estimate = 1.30258136495129,
      std_error = 0.438634194636724,
      lb_ci_alpha05 = 0.349576374985016,
      ub_ci_alpha05 = 2.25558635491756,
      num_df = 1,
      den_df = 12.3141152725094,
      t_statistic = 2.96963023147359,
      p_value = 0.011422794979565
    ),
    FixedEffect(
      id = "SS & SS",
      estimate = 1.05258136495129,
      std_error = 0.438634194636724,
      lb_ci_alpha05 = 0.0995763749850169,
      ub_ci_alpha05 = 2.00558635491755,
      num_df = 0.9999999999999999,
      den_df = 12.3141152725095,
      t_statistic = 2.39967922661167,
      p_value = 0.0330610100059848
    )
  ]
