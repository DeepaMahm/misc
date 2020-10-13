"""
Created by Deepa 28/9/2020
-This code passes inputs from python to autocad
"""
import subprocess
from settings_model import\
    GRAPH_LISP_FILE,\
    GRAPH_SCR_FILE,\
    GRAPH_DWG_FILE,\
    GRAPH_DXF_FILE


def write_src(graph):
    """
    Commands sent to AutoCAD aew
    :param graph:
    :return:
    """
    with open(GRAPH_SCR_FILE, 'w') as outfile:
        outfile.write("open\n" +
                      f"{GRAPH_DWG_FILE}" + "\n"
                      "(setq *LOAD_SECURITY_STATE* (getvar 'SECURELOAD))\n" +
                      '(setvar "SECURELOAD" 0)\n' +
                      rf'(load "{GRAPH_LISP_FILE}")' + "\n"
                      f"{graph}\n" +
                      '(setvar "SECURELOAD" *LOAD_SECURITY_STATE*)\n'
                      "saveas dxf 16\n" +
                      rf"{GRAPH_DXF_FILE}" +"\n"+
                      "quit"
                     )


def run_process(pos, t, h):
    """
    invokes AutoCAD
    :return:
    """


    def iter_to_lisp_string(list_or_tuple):
        """
        Converts python datatypeS list and tuple to AutoCAD datatype
        :param list_or_tuple:
        :return:
        """
        return "(" + " ".join(map(str, list_or_tuple)) + ")"

    pts_string = "'" + iter_to_lisp_string(
        map(iter_to_lisp_string, pos))
    sls_string = "'" + iter_to_lisp_string(t)
    tls_string = "'" + iter_to_lisp_string(h)

    graph_command_string = iter_to_lisp_string(["graph", pts_string, sls_string, tls_string])
    write_src(graph=graph_command_string)

    subprocess.run(["C:\\Program Files\\Autodesk\\AutoCAD 2019\\acad.exe", "/b", GRAPH_SCR_FILE])
    print("Process Finished")


if __name__ == '__main__':

    # test 0
    # pos = [(75, 25, 0), (115, 45, -100), (90, 60, -60), (10, 5, 10), (45, 0, 0), (45, 55, 0), (0, 25, 0)]
    # t = [1, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6]
    # h = [2, 3, 4, 5, 3, 6, 6, 5, 7, 7, 7]

    # test 1
    # pos = [(3811.7929860067793, 3921.5959808650714, 3147.2795932488543), (3833.6976889828165, 3886.893685601582, 3119.4212443771953), (3793.855557193764, 3967.383891235574, 3110.0360293690064), (3788.065269380248, 3975.7538738004264, 3120.2666904991247), (3770.2928284779086, 3976.988863341021, 3137.5194015914503), (3792.3944346580697, 3951.9859148158575, 3156.443865520338), (3782.6068101717997, 3999.438151768058, 3125.1962887976592), (3799.7851726076037, 3887.050489715238, 3155.6162602292407), (3776.3478627153168, 4023.128345468606, 3129.0225614525307), (3772.6697214591654, 4031.914501741826, 3136.9620337750043), (3800.7839566126772, 3938.110956865553, 3141.873462360074), (3794.8778811514694, 3926.0055311214173, 3122.2812944936877), (3791.273199629835, 3871.320262314829, 3109.353255067844), (3765.4097041484306, 4079.667739842486, 3148.2350537014863), (3761.597774045923, 4051.836276243343, 3166.68486384183), (3800.445822305831, 4027.1079819707393, 3231.1124089199425), (3784.3139842340424, 4041.729912236625, 3265.724466879119), (3764.557208930026, 3982.088513936085, 3147.258849734649), (3825.686199454007, 3906.6699108951534, 3114.4824081261227), (3779.1217190177963, 3972.6337446703815, 3159.9848120678607), (3786.152122214654, 3994.370395582202, 3252.2747254553965), (3862.0138968937285, 3770.4195477333183, 3121.370657071021), (3783.9568769506795, 4017.301856019796, 3261.4601377981576), (3819.8003457461573, 4030.3155832055236, 3246.2832162597274), (3802.4351475825206, 3919.306810142977, 3115.0858106483147), (3805.9150176834664, 3960.3784398224557, 3112.519231377173), (3789.7297479014137, 4049.0848742821945, 3234.9998087489626), (3886.029744165818, 3800.8666850877485, 3145.937850311993), (3795.7393286080755, 4038.3934484410374, 3233.1730666988165), (3817.4850931232813, 4085.446216354172, 3332.210175042481), (3858.144453069335, 3840.9101118097924, 3154.8313710295783), (3894.647713045254, 3780.584382312973, 3157.313242525744), (3827.0956132361875, 3930.171042659513, 3140.0016715706), (3845.8396914806794, 3881.543109987466, 3133.0921132916533), (3855.020036166735, 3889.630534333238, 3146.0724493567395), (3802.2331975070965, 4071.069866038269, 3301.476802250279), (3842.360954305007, 4074.347871514186, 3249.7950090614536), (3899.1292904208753, 3772.7133858882053, 3121.232725710983), (3786.0160821308677, 4109.741789730338, 3327.9654162779007), (3784.716703232086, 3959.6892442718645, 3209.9982678223187), (3802.6392136624204, 4004.6568902754025, 3220.808270964723), (3813.3893167221663, 3946.6512419984856, 3152.4748894719023), (3776.9580191293494, 4012.8618258791385, 3126.4060978123857), (3792.9121147869596, 3984.2937132304405, 3163.6819409794552), (3842.2517703579133, 3893.770036410061, 3154.4552865928367), (3799.5446153131074, 3968.7057326911586, 3125.908661649792), (3830.9349040745474, 3920.15899362535, 3178.1068047301064), (3846.1132217746604, 3914.9060912310856, 3152.747211190175), (3840.542456351322, 3939.6311403075265, 3195.3828426681307), (3771.2078037466617, 4040.5515782937478, 3167.1537811826015), (3826.204697462514, 3929.78021130008, 3184.337090087503), (3853.372200021573, 3884.1342705052903, 3174.418438911501), (3822.73449029522, 3943.664123424033, 3119.7840340930666), (3830.8303563600903, 3946.846865555975, 3217.0311120419346), (3903.73490557647, 3875.9580600191334, 3147.286088554153), (3896.2649226233298, 3904.9804707839003, 3134.821791192679), (3787.802378180816, 3993.877478180028, 3169.6654127790534), (3817.0665707568514, 3950.7281049365583, 3193.966346725833), (3810.0687319917574, 3971.8906135993316, 3204.8391186710473), (3818.8604284505645, 3954.4463983985365, 3119.694824694565), (3753.5751777763817, 4061.02387021307, 3182.9775584432878), (3826.82171441451, 3905.8904690718778, 3113.441770671606), (3887.5506949203864, 3889.16378078949, 3102.298971289034), (3854.6943904718405, 3861.057416740752, 3183.4044960821207), (3806.0196197998052, 3963.8580107014113, 3168.7431195493805), (3839.4582916933664, 3879.131589199255, 3068.9935883255475), (3760.6867834615273, 4027.2822489552373, 3106.642355988022), (3848.995859521315, 3915.0507455501997, 3119.7216432360033), (3913.3988496882607, 3862.4685866375553, 3100.5442136341067), (3754.7907651496553, 4019.7125646000536, 3129.5110415238573), (3739.590727611969, 4031.471661205582, 3165.2310005144245), (3731.4689094008027, 4034.872207452946, 3173.9620692807694), (3855.479135286613, 3896.204255088979, 3176.3253010465096), (3751.6820737035905, 4013.406880066007, 3153.293654868611), (3847.259997822387, 3914.7911499237616, 3102.574671642758), (3859.2509659554285, 3886.0378723546187, 3082.244739280329), (3771.272779482301, 4021.5258529640246, 3103.716495556764), (3948.078827729631, 3902.107113543691, 3081.5074091220786), (3861.0378049013507, 3846.9666907472547, 3180.5507156876392), (3861.464398826974, 3821.62598353968, 3155.313078472562), (3816.3638526547043, 3922.402857201051, 3103.561984547092), (3791.1186036752138, 3921.8590580943587, 3128.7537031656825), (3817.4140269420377, 3868.698186212034, 3134.5249117550293), (3871.242951375296, 3867.567071229909, 3171.249048375799), (3793.1884080310488, 3943.582183831022, 3116.969635570735), (3843.512710238644, 3855.409156356263, 3150.0454800214047), (3847.58119728764, 3906.0929652796017, 3144.1487484050494), (3854.2383936551023, 3850.0461860027576, 3169.808905473077), (3782.2498581746668, 3875.657935071885, 3144.468797711151), (3848.2352765211385, 3821.550719237347, 3201.768816234438), (3847.8037239087453, 3829.6583273338383, 3192.3964483664263), (3859.65399954244, 3845.4469711994325, 3148.6555018644917), (3855.590168648663, 3887.7390811136215, 3137.5974772529903), (3876.9187126481843, 3868.5067788625324, 3190.972192983057), (3872.3876772824847, 3847.950516631811, 3272.314031778249), (3801.519561531432, 3901.9861378683804, 3139.3333315437712), (3862.348410066996, 3821.922426052732, 3154.2205047065836), (3862.8068755829217, 3845.455363168514, 3162.0328439749537), (3851.9248479737394, 3867.269553539126, 3164.1086266446864), (3700.571309919157, 4135.730870437051, 3230.3051320523027), (3859.076500048318, 3836.6351508458342, 3224.528289479572), (3895.874909808066, 3837.354853845629, 3229.9319240698715), (3815.872760348203, 3841.138553389788, 3156.219376109788), (3813.483681618057, 3929.8877693681998, 3113.0192810036087), (3860.800027400549, 3885.591969864744, 3216.7551681279783), (3683.494473952708, 4092.662520928703, 3212.5947190912743), (3858.1037833292203, 3867.316695597462, 3174.8593815992995), (3884.5003000591937, 3835.1137562067665, 3147.908407978951), (3808.2881768252023, 4051.217474242231, 3186.3558559151893), (3799.315839566448, 4074.029271290798, 3182.593060617121), (3702.961380598925, 4054.9597883667775, 3186.9100180863697), (3867.287582772614, 3790.4719404215507, 3134.976950977314), (3750.052458448787, 4070.7240442123903, 3177.8650456625505), (3871.4163737682616, 3865.073608329189, 3225.773546259869), (3845.333026629772, 3864.5920714161502, 3140.351096220081), (3809.696320224104, 4121.789045701691, 3174.14277472965), (3869.1669603675423, 3811.520778704624, 3231.7004004508344), (3853.9915969014965, 3882.0024501897965, 3174.626837722615), (3857.583360239801, 3867.397269812611, 3172.343481384163), (3809.612514572221, 3935.5525135952585, 3163.4095697934326), (3839.826359152739, 3884.6779369987194, 3147.1590362607144), (3817.92518938636, 3913.049145440809, 3157.121691567929), (3876.2372743835003, 3871.816952502627, 3155.050027429236), (3805.883342353669, 3852.404694477905, 3152.5123093768), (3902.9216171391504, 3885.168235847699, 3162.920336642412), (3829.1034430760687, 3956.0563996798523, 3189.6196161694497), (3847.645563996087, 3914.0965381452593, 3208.4801984888954), (3860.936530187207, 3872.22821173431, 3119.173235871382), (3874.316892697536, 3851.0556370457016, 3141.621727171725), (3909.127908372891, 3884.769256424036, 3152.1926746717118), (3817.121705090469, 4004.175913353249, 3188.565515847991), (3851.7103679128218, 3892.9772320936995, 3220.829020731408), (3761.178787043623, 4096.190745548501, 3192.4282129366234), (3782.648388490162, 4053.3320718783284, 3261.67772456129), (3807.4514752494283, 3897.4261204608197, 3131.7572766128583), (3881.2048416235466, 3915.2489438257785, 3179.6052803851817), (3855.9654438646667, 3893.696573036907, 3196.40737868692), (3772.25251034154, 4085.0994663731485, 3145.3695543406084), (3810.398924101813, 3910.2823605804106, 3112.1029171911796), (3790.3000980586453, 4089.119819752525, 3255.004696231681), (3776.4887299105276, 4101.067293857237, 3222.5956953067107), (3841.97934615934, 3884.9031471031735, 3180.4120361624855), (3862.359367002015, 3847.750299337114, 3122.8943962957087), (3761.3120175446593, 4090.9267737632563, 3145.040817933143), (3814.688061340792, 3836.729538951181, 3097.97972654178), (3779.847912308957, 4137.596214255755, 3255.9800105830914), (3766.846213056103, 4110.1590003070005, 3212.1210454508296), (3836.3043334112176, 3872.68157432446, 3092.731532614996), (3829.81678655821, 3871.331443792334, 3068.8332263560524), (3850.978793942916, 3833.265206179682, 3090.5755922581066), (3860.876397194665, 3852.350174609542, 3155.4085424752384), (3819.694427198747, 3850.6695752279916, 3078.0887250582514), (3770.9880919340503, 4121.626906054322, 3209.8644431324024), (3754.3200408835187, 4130.236892003849, 3174.4684622048917), (3863.095387596509, 3896.7018119265317, 3042.205444037173), (3744.236181992017, 4178.794889549033, 3175.240195307625), (3875.198722171144, 3865.0104796503347, 3232.739568107086), (3862.2981562042437, 3860.458787616447, 3253.4252590281385), (3872.295514327195, 3902.099648355332, 3140.4970118999213), (3856.0022786430523, 3912.5728141829895, 3155.985267142138), (3841.4303134677357, 3892.849391332046, 3159.0876473537924), (3816.9253761192413, 3962.1402173343686, 3230.735274095251), (3830.2358198195066, 3824.9758620562047, 3158.71365331465), (3801.7151157078797, 3979.2607651158264, 3240.252469625237), (3768.214520415899, 4105.895296096059, 3238.647398037087), (3787.3340825785963, 4112.906997215404, 3224.494038327948), (3845.64436699222, 3843.7311585321295, 3153.6288360345475), (3829.8174576407832, 3921.7930771255506, 3233.1312743202775), (3827.615598322517, 3938.4485040894433, 3251.373769069449), (3779.9357601706124, 4070.7536781341855, 3242.041073806787), (3815.2093901682724, 4041.1013581540133, 3184.479608243281), (3849.882523970212, 3870.300278071189, 3213.573799463567), (3856.286507056754, 3880.947833793804, 3161.4393430826713), (3868.1757729956003, 3877.877682055161, 3122.6953954838573), (3829.1249861447645, 3927.996184798094, 3222.4166808477976), (3829.095516764047, 3909.6292867422926, 3235.429420635511), (3843.3108999058572, 3900.137483863599, 3170.2954940591017), (3863.996103697336, 3877.642333574587, 3157.306389905075), (3819.897577227277, 3883.840412593547, 3265.641162032827), (3806.6827692858474, 3887.29683204906, 3231.039692885559), (3833.4082606526135, 3890.10875706293, 3241.8935190083416), (3837.070415402812, 3950.258519693203, 3182.091514162437), (3839.2244837627422, 3924.959407663965, 3218.7731542266497), (3841.14212922685, 3792.3277736370437, 3098.3000603227524), (3835.533020673161, 3877.439551465943, 3232.499006372257), (3784.8075471745724, 3874.44786814895, 3273.660542937619), (3840.7135938987203, 3863.3686493514415, 3115.9017418325966), (3841.507264209414, 3829.4522478812178, 3100.641212542466), (3823.6901973009603, 3900.5587818654385, 3112.738149555185), (3834.836981221829, 3903.917012025633, 3194.6423797663306), (3833.78803083169, 3852.7180413102305, 3104.403913817961), (3821.7876626076077, 3869.0249355302535, 3202.3722858918677), (3825.6303216712718, 3890.689057432877, 3213.8151597896735), (3838.2954476465466, 3743.5210659710115, 3089.935636376013), (3873.8209330569416, 3857.4311013626793, 3138.562514407697)]
    # t = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 12, 13, 13, 14, 15, 15, 16, 16, 16, 17, 17, 17, 19, 20, 21, 21, 21, 22, 22, 23, 24, 24, 25, 26, 27, 27, 28, 28, 30, 30, 31, 32, 33, 33, 34, 35, 36, 40, 40, 41, 42, 42, 43, 43, 43, 44, 44, 45, 45, 46, 47, 47, 48, 48, 48, 49, 50, 50, 51, 51, 52, 52, 54, 54, 55, 55, 55, 56, 56, 56, 58, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 66, 66, 66, 67, 67, 70, 71, 71, 72, 73, 75, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 85, 86, 86, 86, 87, 87, 88, 89, 90, 90, 91, 93, 94, 94, 94, 95, 95, 95, 97, 97, 98, 99, 99, 99, 100, 100, 101, 101, 102, 102, 104, 105, 105, 106, 107, 108, 109, 109, 109, 110, 110, 118, 118, 119, 119, 120, 120, 121, 121, 121, 122, 123, 123, 124, 124, 125, 125, 126, 126, 126, 127, 128, 129, 130, 131, 132, 133, 133, 134, 134, 136, 136, 138, 139, 140, 140, 141, 141, 142, 143, 144, 145, 145, 146, 147, 148, 149, 149, 154, 157, 157, 158, 159, 159, 160, 160, 161, 161, 162, 162, 163, 163, 164, 165, 165, 166, 168, 168, 168, 169, 172, 172, 173, 173, 175, 176, 177, 178, 179, 179, 179, 180, 180, 183, 183, 184, 184, 187, 188, 189, 192]
    # h = [2, 6, 8, 187, 7, 18, 85, 5, 7, 26, 11, 18, 20, 120, 9, 13, 31, 10, 15, 14, 26, 45, 19, 20, 25, 19, 25, 144, 70, 113, 24, 29, 41, 23, 36, 134, 34, 74, 23, 24, 164, 32, 112, 29, 27, 37, 139, 53, 29, 170, 31, 32, 36, 37, 35, 38, 35, 42, 93, 84, 39, 41, 58, 59, 46, 65, 44, 46, 50, 57, 65, 47, 52, 60, 51, 177, 49, 53, 87, 54, 57, 138, 58, 64, 64, 73, 162, 181, 63, 69, 151, 63, 78, 159, 59, 65, 62, 68, 71, 133, 68, 75, 68, 69, 123, 75, 76, 148, 70, 77, 74, 72, 74, 111, 191, 76, 84, 91, 84, 103, 92, 83, 104, 189, 85, 89, 96, 89, 96, 104, 90, 98, 167, 88, 92, 93, 103, 91, 97, 103, 173, 101, 107, 178, 102, 114, 158, 98, 112, 119, 107, 115, 172, 106, 165, 102, 105, 114, 117, 152, 114, 132, 111, 108, 115, 110, 113, 171, 113, 116, 129, 137, 125, 142, 122, 135, 122, 128, 143, 124, 129, 151, 135, 163, 130, 128, 127, 131, 182, 132, 143, 130, 136, 171, 137, 138, 147, 140, 141, 137, 142, 144, 148, 141, 146, 146, 153, 190, 150, 154, 150, 152, 153, 153, 149, 152, 155, 156, 158, 181, 176, 160, 161, 174, 182, 167, 182, 164, 175, 167, 174, 169, 166, 170, 171, 169, 175, 176, 181, 180, 185, 177, 195, 190, 183, 192, 187, 180, 185, 186, 186, 192, 185, 193, 188, 194, 188, 191, 191, 193]

    # test 2
    pos = [(-3811.7929860067793, -3921.5959808650714, -3147.2795932488543), (-3833.6976889828165, -3886.893685601582, -3119.4212443771953), (-3793.855557193764, -3967.383891235574, -3110.0360293690064), (-3788.065269380248, -3975.7538738004264, -3120.2666904991247), (-3770.2928284779086, -3976.988863341021, -3137.5194015914503), (-3792.3944346580697, -3951.9859148158575, -3156.443865520338), (-3782.6068101717997, -3999.438151768058, -3125.1962887976592), (-3799.7851726076037, -3887.050489715238, -3155.6162602292407), (-3776.3478627153168, -4023.128345468606, -3129.0225614525307), (-3772.6697214591654, -4031.914501741826, -3136.9620337750043), (-3800.7839566126772, -3938.110956865553, -3141.873462360074), (-3794.8778811514694, -3926.0055311214173, -3122.2812944936877), (-3791.273199629835, -3871.320262314829, -3109.353255067844), (-3765.4097041484306, -4079.667739842486, -3148.2350537014863), (-3761.597774045923, -4051.836276243343, -3166.68486384183), (-3800.445822305831, -4027.1079819707393, -3231.1124089199425), (-3784.3139842340424, -4041.729912236625, -3265.724466879119), (-3764.557208930026, -3982.088513936085, -3147.258849734649), (-3825.686199454007, -3906.6699108951534, -3114.4824081261227), (-3779.1217190177963, -3972.6337446703815, -3159.9848120678607), (-3786.152122214654, -3994.370395582202, -3252.2747254553965), (-3862.0138968937285, -3770.4195477333183, -3121.370657071021), (-3783.9568769506795, -4017.301856019796, -3261.4601377981576), (-3819.8003457461573, -4030.3155832055236, -3246.2832162597274), (-3802.4351475825206, -3919.306810142977, -3115.0858106483147), (-3805.9150176834664, -3960.3784398224557, -3112.519231377173), (-3789.7297479014137, -4049.0848742821945, -3234.9998087489626), (-3886.029744165818, -3800.8666850877485, -3145.937850311993), (-3795.7393286080755, -4038.3934484410374, -3233.1730666988165), (-3817.4850931232813, -4085.446216354172, -3332.210175042481), (-3858.144453069335, -3840.9101118097924, -3154.8313710295783), (-3894.647713045254, -3780.584382312973, -3157.313242525744), (-3827.0956132361875, -3930.171042659513, -3140.0016715706), (-3845.8396914806794, -3881.543109987466, -3133.0921132916533), (-3855.020036166735, -3889.630534333238, -3146.0724493567395), (-3802.2331975070965, -4071.069866038269, -3301.476802250279), (-3842.360954305007, -4074.347871514186, -3249.7950090614536), (-3899.1292904208753, -3772.7133858882053, -3121.232725710983), (-3786.0160821308677, -4109.741789730338, -3327.9654162779007), (-3784.716703232086, -3959.6892442718645, -3209.9982678223187), (-3802.6392136624204, -4004.6568902754025, -3220.808270964723), (-3813.3893167221663, -3946.6512419984856, -3152.4748894719023), (-3776.9580191293494, -4012.8618258791385, -3126.4060978123857), (-3792.9121147869596, -3984.2937132304405, -3163.6819409794552), (-3842.2517703579133, -3893.770036410061, -3154.4552865928367), (-3799.5446153131074, -3968.7057326911586, -3125.908661649792), (-3830.9349040745474, -3920.15899362535, -3178.1068047301064), (-3846.1132217746604, -3914.9060912310856, -3152.747211190175), (-3840.542456351322, -3939.6311403075265, -3195.3828426681307), (-3771.2078037466617, -4040.5515782937478, -3167.1537811826015), (-3826.204697462514, -3929.78021130008, -3184.337090087503), (-3853.372200021573, -3884.1342705052903, -3174.418438911501), (-3822.73449029522, -3943.664123424033, -3119.7840340930666), (-3830.8303563600903, -3946.846865555975, -3217.0311120419346), (-3903.73490557647, -3875.9580600191334, -3147.286088554153), (-3896.2649226233298, -3904.9804707839003, -3134.821791192679), (-3787.802378180816, -3993.877478180028, -3169.6654127790534), (-3817.0665707568514, -3950.7281049365583, -3193.966346725833), (-3810.0687319917574, -3971.8906135993316, -3204.8391186710473), (-3818.8604284505645, -3954.4463983985365, -3119.694824694565), (-3753.5751777763817, -4061.02387021307, -3182.9775584432878), (-3826.82171441451, -3905.8904690718778, -3113.441770671606), (-3887.5506949203864, -3889.16378078949, -3102.298971289034), (-3854.6943904718405, -3861.057416740752, -3183.4044960821207), (-3806.0196197998052, -3963.8580107014113, -3168.7431195493805), (-3839.4582916933664, -3879.131589199255, -3068.9935883255475), (-3760.6867834615273, -4027.2822489552373, -3106.642355988022), (-3848.995859521315, -3915.0507455501997, -3119.7216432360033), (-3913.3988496882607, -3862.4685866375553, -3100.5442136341067), (-3754.7907651496553, -4019.7125646000536, -3129.5110415238573), (-3739.590727611969, -4031.471661205582, -3165.2310005144245), (-3731.4689094008027, -4034.872207452946, -3173.9620692807694), (-3855.479135286613, -3896.204255088979, -3176.3253010465096), (-3751.6820737035905, -4013.406880066007, -3153.293654868611), (-3847.259997822387, -3914.7911499237616, -3102.574671642758), (-3859.2509659554285, -3886.0378723546187, -3082.244739280329), (-3771.272779482301, -4021.5258529640246, -3103.716495556764), (-3948.078827729631, -3902.107113543691, -3081.5074091220786), (-3861.0378049013507, -3846.9666907472547, -3180.5507156876392), (-3861.464398826974, -3821.62598353968, -3155.313078472562), (-3816.3638526547043, -3922.402857201051, -3103.561984547092), (-3791.1186036752138, -3921.8590580943587, -3128.7537031656825), (-3817.4140269420377, -3868.698186212034, -3134.5249117550293), (-3871.242951375296, -3867.567071229909, -3171.249048375799), (-3793.1884080310488, -3943.582183831022, -3116.969635570735), (-3843.512710238644, -3855.409156356263, -3150.0454800214047), (-3847.58119728764, -3906.0929652796017, -3144.1487484050494), (-3854.2383936551023, -3850.0461860027576, -3169.808905473077), (-3782.2498581746668, -3875.657935071885, -3144.468797711151), (-3848.2352765211385, -3821.550719237347, -3201.768816234438), (-3847.8037239087453, -3829.6583273338383, -3192.3964483664263), (-3859.65399954244, -3845.4469711994325, -3148.6555018644917), (-3855.590168648663, -3887.7390811136215, -3137.5974772529903), (-3876.9187126481843, -3868.5067788625324, -3190.972192983057), (-3872.3876772824847, -3847.950516631811, -3272.314031778249), (-3801.519561531432, -3901.9861378683804, -3139.3333315437712), (-3862.348410066996, -3821.922426052732, -3154.2205047065836), (-3862.8068755829217, -3845.455363168514, -3162.0328439749537), (-3851.9248479737394, -3867.269553539126, -3164.1086266446864), (-3700.571309919157, -4135.730870437051, -3230.3051320523027), (-3859.076500048318, -3836.6351508458342, -3224.528289479572), (-3895.874909808066, -3837.354853845629, -3229.9319240698715), (-3815.872760348203, -3841.138553389788, -3156.219376109788), (-3813.483681618057, -3929.8877693681998, -3113.0192810036087), (-3860.800027400549, -3885.591969864744, -3216.7551681279783), (-3683.494473952708, -4092.662520928703, -3212.5947190912743), (-3858.1037833292203, -3867.316695597462, -3174.8593815992995), (-3884.5003000591937, -3835.1137562067665, -3147.908407978951), (-3808.2881768252023, -4051.217474242231, -3186.3558559151893), (-3799.315839566448, -4074.029271290798, -3182.593060617121), (-3702.961380598925, -4054.9597883667775, -3186.9100180863697), (-3867.287582772614, -3790.4719404215507, -3134.976950977314), (-3750.052458448787, -4070.7240442123903, -3177.8650456625505), (-3871.4163737682616, -3865.073608329189, -3225.773546259869), (-3845.333026629772, -3864.5920714161502, -3140.351096220081), (-3809.696320224104, -4121.789045701691, -3174.14277472965), (-3869.1669603675423, -3811.520778704624, -3231.7004004508344), (-3853.9915969014965, -3882.0024501897965, -3174.626837722615), (-3857.583360239801, -3867.397269812611, -3172.343481384163), (-3809.612514572221, -3935.5525135952585, -3163.4095697934326), (-3839.826359152739, -3884.6779369987194, -3147.1590362607144), (-3817.92518938636, -3913.049145440809, -3157.121691567929), (-3876.2372743835003, -3871.816952502627, -3155.050027429236), (-3805.883342353669, -3852.404694477905, -3152.5123093768), (-3902.9216171391504, -3885.168235847699, -3162.920336642412), (-3829.1034430760687, -3956.0563996798523, -3189.6196161694497), (-3847.645563996087, -3914.0965381452593, -3208.4801984888954), (-3860.936530187207, -3872.22821173431, -3119.173235871382), (-3874.316892697536, -3851.0556370457016, -3141.621727171725), (-3909.127908372891, -3884.769256424036, -3152.1926746717118), (-3817.121705090469, -4004.175913353249, -3188.565515847991), (-3851.7103679128218, -3892.9772320936995, -3220.829020731408), (-3761.178787043623, -4096.190745548501, -3192.4282129366234), (-3782.648388490162, -4053.3320718783284, -3261.67772456129), (-3807.4514752494283, -3897.4261204608197, -3131.7572766128583), (-3881.2048416235466, -3915.2489438257785, -3179.6052803851817), (-3855.9654438646667, -3893.696573036907, -3196.40737868692), (-3772.25251034154, -4085.0994663731485, -3145.3695543406084), (-3810.398924101813, -3910.2823605804106, -3112.1029171911796), (-3790.3000980586453, -4089.119819752525, -3255.004696231681), (-3776.4887299105276, -4101.067293857237, -3222.5956953067107), (-3841.97934615934, -3884.9031471031735, -3180.4120361624855), (-3862.359367002015, -3847.750299337114, -3122.8943962957087), (-3761.3120175446593, -4090.9267737632563, -3145.040817933143), (-3814.688061340792, -3836.729538951181, -3097.97972654178), (-3779.847912308957, -4137.596214255755, -3255.9800105830914), (-3766.846213056103, -4110.1590003070005, -3212.1210454508296), (-3836.3043334112176, -3872.68157432446, -3092.731532614996), (-3829.81678655821, -3871.331443792334, -3068.8332263560524), (-3850.978793942916, -3833.265206179682, -3090.5755922581066), (-3860.876397194665, -3852.350174609542, -3155.4085424752384), (-3819.694427198747, -3850.6695752279916, -3078.0887250582514), (-3770.9880919340503, -4121.626906054322, -3209.8644431324024), (-3754.3200408835187, -4130.236892003849, -3174.4684622048917), (-3863.095387596509, -3896.7018119265317, -3042.205444037173), (-3744.236181992017, -4178.794889549033, -3175.240195307625), (-3875.198722171144, -3865.0104796503347, -3232.739568107086), (-3862.2981562042437, -3860.458787616447, -3253.4252590281385), (-3872.295514327195, -3902.099648355332, -3140.4970118999213), (-3856.0022786430523, -3912.5728141829895, -3155.985267142138), (-3841.4303134677357, -3892.849391332046, -3159.0876473537924), (-3816.9253761192413, -3962.1402173343686, -3230.735274095251), (-3830.2358198195066, -3824.9758620562047, -3158.71365331465), (-3801.7151157078797, -3979.2607651158264, -3240.252469625237), (-3768.214520415899, -4105.895296096059, -3238.647398037087), (-3787.3340825785963, -4112.906997215404, -3224.494038327948), (-3845.64436699222, -3843.7311585321295, -3153.6288360345475), (-3829.8174576407832, -3921.7930771255506, -3233.1312743202775), (-3827.615598322517, -3938.4485040894433, -3251.373769069449), (-3779.9357601706124, -4070.7536781341855, -3242.041073806787), (-3815.2093901682724, -4041.1013581540133, -3184.479608243281), (-3849.882523970212, -3870.300278071189, -3213.573799463567), (-3856.286507056754, -3880.947833793804, -3161.4393430826713), (-3868.1757729956003, -3877.877682055161, -3122.6953954838573), (-3829.1249861447645, -3927.996184798094, -3222.4166808477976), (-3829.095516764047, -3909.6292867422926, -3235.429420635511), (-3843.3108999058572, -3900.137483863599, -3170.2954940591017), (-3863.996103697336, -3877.642333574587, -3157.306389905075), (-3819.897577227277, -3883.840412593547, -3265.641162032827), (-3806.6827692858474, -3887.29683204906, -3231.039692885559), (-3833.4082606526135, -3890.10875706293, -3241.8935190083416), (-3837.070415402812, -3950.258519693203, -3182.091514162437), (-3839.2244837627422, -3924.959407663965, -3218.7731542266497), (-3841.14212922685, -3792.3277736370437, -3098.3000603227524), (-3835.533020673161, -3877.439551465943, -3232.499006372257), (-3784.8075471745724, -3874.44786814895, -3273.660542937619), (-3840.7135938987203, -3863.3686493514415, -3115.9017418325966), (-3841.507264209414, -3829.4522478812178, -3100.641212542466), (-3823.6901973009603, -3900.5587818654385, -3112.738149555185), (-3834.836981221829, -3903.917012025633, -3194.6423797663306), (-3833.78803083169, -3852.7180413102305, -3104.403913817961), (-3821.7876626076077, -3869.0249355302535, -3202.3722858918677), (-3825.6303216712718, -3890.689057432877, -3213.8151597896735), (-3838.2954476465466, -3743.5210659710115, -3089.935636376013), (-3873.8209330569416, -3857.4311013626793, -3138.562514407697)]
    t = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 12, 13, 13, 14, 15, 15, 16, 16, 16, 17, 17, 17, 19, 20, 21, 21, 21, 22, 22, 23, 24, 24, 25, 26, 27, 27, 28, 28, 30, 30, 31, 32, 33, 33, 34, 35, 36, 40, 40, 41, 42, 42, 43, 43, 43, 44, 44, 45, 45, 46, 47, 47, 48, 48, 48, 49, 50, 50, 51, 51, 52, 52, 54, 54, 55, 55, 55, 56, 56, 56, 58, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 66, 66, 66, 67, 67, 70, 71, 71, 72, 73, 75, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 85, 86, 86, 86, 87, 87, 88, 89, 90, 90, 91, 93, 94, 94, 94, 95, 95, 95, 97, 97, 98, 99, 99, 99, 100, 100, 101, 101, 102, 102, 104, 105, 105, 106, 107, 108, 109, 109, 109, 110, 110, 118, 118, 119, 119, 120, 120, 121, 121, 121, 122, 123, 123, 124, 124, 125, 125, 126, 126, 126, 127, 128, 129, 130, 131, 132, 133, 133, 134, 134, 136, 136, 138, 139, 140, 140, 141, 141, 142, 143, 144, 145, 145, 146, 147, 148, 149, 149, 154, 157, 157, 158, 159, 159, 160, 160, 161, 161, 162, 162, 163, 163, 164, 165, 165, 166, 168, 168, 168, 169, 172, 172, 173, 173, 175, 176, 177, 178, 179, 179, 179, 180, 180, 183, 183, 184, 184, 187, 188, 189, 192]
    h = [2, 6, 8, 187, 7, 18, 85, 5, 7, 26, 11, 18, 20, 120, 9, 13, 31, 10, 15, 14, 26, 45, 19, 20, 25, 19, 25, 144, 70, 113, 24, 29, 41, 23, 36, 134, 34, 74, 23, 24, 164, 32, 112, 29, 27, 37, 139, 53, 29, 170, 31, 32, 36, 37, 35, 38, 35, 42, 93, 84, 39, 41, 58, 59, 46, 65, 44, 46, 50, 57, 65, 47, 52, 60, 51, 177, 49, 53, 87, 54, 57, 138, 58, 64, 64, 73, 162, 181, 63, 69, 151, 63, 78, 159, 59, 65, 62, 68, 71, 133, 68, 75, 68, 69, 123, 75, 76, 148, 70, 77, 74, 72, 74, 111, 191, 76, 84, 91, 84, 103, 92, 83, 104, 189, 85, 89, 96, 89, 96, 104, 90, 98, 167, 88, 92, 93, 103, 91, 97, 103, 173, 101, 107, 178, 102, 114, 158, 98, 112, 119, 107, 115, 172, 106, 165, 102, 105, 114, 117, 152, 114, 132, 111, 108, 115, 110, 113, 171, 113, 116, 129, 137, 125, 142, 122, 135, 122, 128, 143, 124, 129, 151, 135, 163, 130, 128, 127, 131, 182, 132, 143, 130, 136, 171, 137, 138, 147, 140, 141, 137, 142, 144, 148, 141, 146, 146, 153, 190, 150, 154, 150, 152, 153, 153, 149, 152, 155, 156, 158, 181, 176, 160, 161, 174, 182, 167, 182, 164, 175, 167, 174, 169, 166, 170, 171, 169, 175, 176, 181, 180, 185, 177, 195, 190, 183, 192, 187, 180, 185, 186, 186, 192, 185, 193, 188, 194, 188, 191, 191, 193]
    run_process(pos, t, h)
