# MTL exps with different λ: 0, ∞, 2, 1, 1/3, 1/8, 1/24
random_seeds = [1234, 12345, 123, 111, 1111]
# random_seeds = [12345, 123, 111, 1111]
all_valid_exps = [
    # random seed 1234
    [
        "expstlq001",
        "expstlet001",
        "expmtl002",
        "expmtl001",
        "expmtl003",
        "expmtl004",
        "expmtl005",
    ],
    # random seed 12345
    [
        "expstlq202",
        "expstlet003",
        "expmtl202",
        "expmtl201",
        "expmtl203",
        "expmtl204",
        "expmtl205",
    ],
    # random seed 123
    [
        "expstlq203",
        "expstlet002",
        "expmtl302",
        "expmtl301",
        "expmtl303",
        "expmtl304",
        "expmtl305",
    ],
    # random seed 111
    [
        "expstlq204",
        "expstlet004",
        "expmtl402",
        "expmtl401",
        "expmtl403",
        "expmtl404",
        "expmtl405",
    ],
    # random seed 1111
    [
        "expstlq205",
        "expstlet005",
        "expmtl502",
        "expmtl501",
        "expmtl503",
        "expmtl504",
        "expmtl505",
    ],
]
all_weight_files = [
    # random seed 1234
    [
        "07_April_202311_52AM_model.pth",
        "09_April_202303_02AM_model.pth",
        "09_April_202303_57PM_model.pth",
        "12_April_202305_24PM_model.pth",
        "12_April_202306_35PM_model.pth",
        "14_April_202302_40PM_model.pth",
        "14_April_202304_16PM_model.pth",
    ],
    # random seed 12345
    [
        "21_May_202403_13PM_model.pth",
        "24_April_202309_24PM_model.pth",
        "19_May_202401_01AM_model.pth",
        "18_May_202411_27PM_model.pth",
        "17_May_202404_59PM_model.pth",
        "18_May_202411_23PM_model.pth",
        "19_May_202401_06AM_model.pth",
    ],
    # random seed 123
    [
        "21_May_202408_46PM_model.pth",
        "11_April_202303_45AM_model.pth",
        "19_May_202409_57PM_model.pth",
        "19_May_202409_55PM_model.pth",
        "18_May_202405_18PM_model.pth",
        "19_May_202411_57PM_model.pth",
        "19_May_202411_58PM_model.pth",
    ],
    # random seed 111
    [
        "23_May_202403_18PM_model.pth",
        "24_April_202310_21PM_model.pth",
        "21_May_202409_11PM_model.pth",
        "21_May_202409_11PM_model.pth",
        "21_May_202409_33PM_model.pth",
        "21_May_202411_50PM_model.pth",
        "21_May_202411_51PM_model.pth",
    ],
    # random seed 1111
    [
        "23_May_202403_32PM_model.pth",
        "26_April_202303_04PM_model.pth",
        "23_May_202410_34PM_model.pth",
        "23_May_202409_34PM_model.pth",
        "24_May_202402_13AM_model.pth",
        "24_May_202402_33AM_model.pth",
        "24_May_202402_13AM_model.pth",
    ],
]
all_loss_weights = [
    [1, 0],
    [0, 1],
    [0.33, 0.66],
    [0.5, 0.5],
    [0.75, 0.25],
    [0.88, 0.11],
    [0.96, 0.04],
]
all_train_exps = [
    # random seed 1234
    [
        "expstlqtrain001",
        "expstlettrain001",
        "expmtltrain002",
        "expmtltrain001",
        "expmtltrain003",
        "expmtltrain004",
        "expmtltrain005",
    ],
    # random seed 12345
    [
        "expstlqtrain202",
        "expstlettrain003",
        "expmtltrain202",
        "expmtltrain201",
        "expmtltrain203",
        "expmtltrain204",
        "expmtltrain205",
    ],
    # random seed 123
    [
        "expstlqtrain203",
        "expstlettrain002",
        "expmtltrain302",
        "expmtltrain301",
        "expmtltrain303",
        "expmtltrain304",
        "expmtltrain305",
    ],
    # random seed 111
    [
        "expstlqtrain204",
        "expstlettrain004",
        "expmtltrain402",
        "expmtltrain401",
        "expmtltrain403",
        "expmtltrain404",
        "expmtltrain405",
    ],
    # random seed 1111
    [
        "expstlqtrain205",
        "expstlettrain005",
        "expmtltrain502",
        "expmtltrain501",
        "expmtltrain503",
        "expmtltrain504",
        "expmtltrain505",
    ],
]
