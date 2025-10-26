import itertools
from .transforms_ss import *
from torchvision.transforms import Compose
from dataload.datasets import Charades, HMDB51, UCF101, AnimalKingdom, kinetics400
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from catalyst.data.sampler import DistributedSamplerWrapper
from utils.tools import fixed_random_seed


# Video data restriction class
class LimitDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


# Data set management class
class DataManager():
    def __init__(self, args, path):
        fixed_random_seed(args.seed)
        self.path = path
        self.distributed = args.distributed

        self.file_number = args.data.file_number
        self.num_workers = args.data.num_workers
        self.dataset = args.data.dataset
        self.sample = args.data.sample

        self.total_length = args.iterate.total_length
        self.batch_size = args.iterate.batch_size

        self.zero = args.expt.zero
        self.few = args.expt.few

    # Checks whether the data set is within the selected range
    def _check(self, ):
        datasets_list = ["charades", 'hmdb51', 'ucf101', "animalkingdom", "kinetics400"]
        if (self.dataset not in datasets_list):
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")

    # Return number of categories
    def get_num_classes(self, ):
        self._check()
        if (self.dataset == "charades"):
            return 157
        elif (self.dataset == "hmdb51"):
            return 51
        elif (self.dataset == "ucf101"):
            return 101
        elif (self.dataset == "animalkingdom"):
            return 140
        elif (self.dataset == "kinetics400"):
            return 400

    # Get class dictionary
    def get_act_dict(self, ):
        self._check()
        animalkingdom_dict = {'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4,
                              'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8,
                              'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12,
                              'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17,
                              'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21,
                              'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25,
                              'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28,
                              'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31,
                              'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34,
                              'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39,
                              'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43,
                              'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48,
                              'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52,
                              'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56,
                              'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60,
                              'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64,
                              'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69,
                              'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73,
                              'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77,
                              'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81,
                              'Performing allo-grooming': 82, 'Performing allo-preening': 83,
                              'Performing copulatory mounting': 84, 'Performing sexual exploration': 85,
                              'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89,
                              'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94,
                              'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99,
                              'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103,
                              'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107,
                              'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111,
                              'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115,
                              'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119,
                              'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123,
                              'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127,
                              'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131,
                              'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136,
                              'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139}
        charades_dict = {'Holding some clothes': 0, 'Putting clothes somewhere': 1,
                         'Taking some clothes from somewhere': 2, 'Throwing clothes somewhere': 3,
                         'Tidying some clothes': 4, 'Washing some clothes': 5, 'Closing a door': 6, 'Fixing a door': 7,
                         'Opening a door': 8, 'Putting something on a table': 9, 'Sitting on a table': 10,
                         'Sitting at a table': 11, 'Tidying up a table': 12, 'Washing a table': 13,
                         'Working at a table': 14, 'Holding a phone/camera': 15, 'Playing with a phone/camera': 16,
                         'Putting a phone/camera somewhere': 17, 'Taking a phone/camera from somewhere': 18,
                         'Talking on a phone/camera': 19, 'Holding a bag': 20, 'Opening a bag': 21,
                         'Putting a bag somewhere': 22, 'Taking a bag from somewhere': 23,
                         'Throwing a bag somewhere': 24, 'Closing a book': 25, 'Holding a book': 26,
                         'Opening a book': 27, 'Putting a book somewhere': 28, 'Smiling at a book': 29,
                         'Taking a book from somewhere': 30, 'Throwing a book somewhere': 31,
                         'Watching/Reading/Looking at a book': 32, 'Holding a towel/s': 33,
                         'Putting a towel/s somewhere': 34, 'Taking a towel/s from somewhere': 35,
                         'Throwing a towel/s somewhere': 36, 'Tidying up a towel/s': 37,
                         'Washing something with a towel': 38, 'Closing a box': 39, 'Holding a box': 40,
                         'Opening a box': 41, 'Putting a box somewhere': 42, 'Taking a box from somewhere': 43,
                         'Taking something from a box': 44, 'Throwing a box somewhere': 45, 'Closing a laptop': 46,
                         'Holding a laptop': 47, 'Opening a laptop': 48, 'Putting a laptop somewhere': 49,
                         'Taking a laptop from somewhere': 50, 'Watching a laptop or something on a laptop': 51,
                         'Working/Playing on a laptop': 52, 'Holding a shoe/shoes': 53, 'Putting shoes somewhere': 54,
                         'Putting on shoe/shoes': 55, 'Taking shoes from somewhere': 56, 'Taking off some shoes': 57,
                         'Throwing shoes somewhere': 58, 'Sitting in a chair': 59, 'Standing on a chair': 60,
                         'Holding some food': 61, 'Putting some food somewhere': 62, 'Taking food from somewhere': 63,
                         'Throwing food somewhere': 64, 'Eating a sandwich': 65, 'Making a sandwich': 66,
                         'Holding a sandwich': 67, 'Putting a sandwich somewhere': 68,
                         'Taking a sandwich from somewhere': 69, 'Holding a blanket': 70,
                         'Putting a blanket somewhere': 71, 'Snuggling with a blanket': 72,
                         'Taking a blanket from somewhere': 73, 'Throwing a blanket somewhere': 74,
                         'Tidying up a blanket/s': 75, 'Holding a pillow': 76, 'Putting a pillow somewhere': 77,
                         'Snuggling with a pillow': 78, 'Taking a pillow from somewhere': 79,
                         'Throwing a pillow somewhere': 80, 'Putting something on a shelf': 81,
                         'Tidying a shelf or something on a shelf': 82, 'Reaching for and grabbing a picture': 83,
                         'Holding a picture': 84, 'Laughing at a picture': 85, 'Putting a picture somewhere': 86,
                         'Taking a picture of something': 87, 'Watching/looking at a picture': 88,
                         'Closing a window': 89, 'Opening a window': 90, 'Washing a window': 91,
                         'Watching/Looking outside of a window': 92, 'Holding a mirror': 93, 'Smiling in a mirror': 94,
                         'Washing a mirror': 95, 'Watching something/someone/themselves in a mirror': 96,
                         'Walking through a doorway': 97, 'Holding a broom': 98, 'Putting a broom somewhere': 99,
                         'Taking a broom from somewhere': 100, 'Throwing a broom somewhere': 101,
                         'Tidying up with a broom': 102, 'Fixing a light': 103, 'Turning on a light': 104,
                         'Turning off a light': 105, 'Drinking from a cup/glass/bottle': 106,
                         'Holding a cup/glass/bottle of something': 107,
                         'Pouring something into a cup/glass/bottle': 108, 'Putting a cup/glass/bottle somewhere': 109,
                         'Taking a cup/glass/bottle from somewhere': 110, 'Washing a cup/glass/bottle': 111,
                         'Closing a closet/cabinet': 112, 'Opening a closet/cabinet': 113,
                         'Tidying up a closet/cabinet': 114, 'Someone is holding a paper/notebook': 115,
                         'Putting their paper/notebook somewhere': 116, 'Taking paper/notebook from somewhere': 117,
                         'Holding a dish': 118, 'Putting a dish/es somewhere': 119,
                         'Taking a dish/es from somewhere': 120, 'Wash a dish/dishes': 121,
                         'Lying on a sofa/couch': 122, 'Sitting on sofa/couch': 123, 'Lying on the floor': 124,
                         'Sitting on the floor': 125, 'Throwing something on the floor': 126,
                         'Tidying something on the floor': 127, 'Holding some medicine': 128,
                         'Taking/consuming some medicine': 129, 'Putting groceries somewhere': 130,
                         'Laughing at television': 131, 'Watching television': 132, 'Someone is awakening in bed': 133,
                         'Lying on a bed': 134, 'Sitting in a bed': 135, 'Fixing a vacuum': 136,
                         'Holding a vacuum': 137, 'Taking a vacuum from somewhere': 138, 'Washing their hands': 139,
                         'Fixing a doorknob': 140, 'Grasping onto a doorknob': 141, 'Closing a refrigerator': 142,
                         'Opening a refrigerator': 143, 'Fixing their hair': 144, 'Working on paper/notebook': 145,
                         'Someone is awakening somewhere': 146, 'Someone is cooking something': 147,
                         'Someone is dressing': 148, 'Someone is laughing': 149, 'Someone is running somewhere': 150,
                         'Someone is going from standing to sitting': 151, 'Someone is smiling': 152,
                         'Someone is sneezing': 153, 'Someone is standing up from somewhere': 154,
                         'Someone is undressing': 155, 'Someone is eating something': 156}
        hmdb51_dict = {'brush_hair': 0, 'cartwheel': 1, 'catch': 2, 'chew': 3, 'clap': 4, 'climb': 5, 'climb_stairs': 6,
                       'dive': 7, 'draw_sword': 8, 'dribble': 9, 'drink': 10, 'eat': 11, 'fall_floor': 12,
                       'fencing': 13, 'flic_flac': 14, 'golf': 15, 'handstand': 16, 'hit': 17, 'hug': 18, 'jump': 19,
                       'kick': 20, 'kick_ball': 21, 'kiss': 22, 'laugh': 23, 'pick': 24, 'pour': 25, 'pullup': 26,
                       'punch': 27, 'push': 28, 'pushup': 29, 'ride_bike': 30, 'ride_horse': 31, 'run': 32,
                       'shake_hands': 33, 'shoot_ball': 34, 'shoot_bow': 35, 'shoot_gun': 36, 'sit': 37, 'situp': 38,
                       'smile': 39, 'smoke': 40, 'somersault': 41, 'stand': 42, 'swing_baseball': 43, 'sword': 44,
                       'sword_exercise': 45, 'talk': 46, 'throw': 47, 'turn': 48, 'walk': 49, 'wave': 50}
        ucf101_dict = {'ApplyEyeMakeup': 0, 'ApplyLipstick': 1, 'Archery': 2, 'BabyCrawling': 3, 'BalanceBeam': 4,
                       'BandMarching': 5, 'BaseballPitch': 6, 'Basketball': 7, 'BasketballDunk': 8, 'BenchPress': 9,
                       'Biking': 10, 'Billiards': 11, 'BlowDryHair': 12, 'BlowingCandles': 13, 'BodyWeightSquats': 14,
                       'Bowling': 15, 'BoxingPunchingBag': 16, 'BoxingSpeedBag': 17, 'BreastStroke': 18,
                       'BrushingTeeth': 19, 'CleanAndJerk': 20, 'CliffDiving': 21, 'CricketBowling': 22,
                       'CricketShot': 23, 'CuttingInKitchen': 24, 'Diving': 25, 'Drumming': 26, 'Fencing': 27,
                       'FieldHockeyPenalty': 28, 'FloorGymnastics': 29, 'FrisbeeCatch': 30, 'FrontCrawl': 31,
                       'GolfSwing': 32, 'Haircut': 33, 'Hammering': 34, 'HammerThrow': 35, 'HandstandPushups': 36,
                       'HandstandWalking': 37, 'HeadMassage': 38, 'HighJump': 39, 'HorseRace': 40, 'HorseRiding': 41,
                       'HulaHoop': 42, 'IceDancing': 43, 'JavelinThrow': 44, 'JugglingBalls': 45, 'JumpingJack': 46,
                       'JumpRope': 47, 'Kayaking': 48, 'Knitting': 49, 'LongJump': 50, 'Lunges': 51,
                       'MilitaryParade': 52, 'Mixing': 53, 'MoppingFloor': 54, 'Nunchucks': 55, 'ParallelBars': 56,
                       'PizzaTossing': 57, 'PlayingCello': 58, 'PlayingDaf': 59, 'PlayingDhol': 60, 'PlayingFlute': 61,
                       'PlayingGuitar': 62, 'PlayingPiano': 63, 'PlayingSitar': 64, 'PlayingTabla': 65,
                       'PlayingViolin': 66, 'PoleVault': 67, 'PommelHorse': 68, 'PullUps': 69, 'Punch': 70,
                       'PushUps': 71, 'Rafting': 72, 'RockClimbingIndoor': 73, 'RopeClimbing': 74, 'Rowing': 75,
                       'SalsaSpin': 76, 'ShavingBeard': 77, 'Shotput': 78, 'SkateBoarding': 79, 'Skiing': 80,
                       'Skijet': 81, 'SkyDiving': 82, 'SoccerJuggling': 83, 'SoccerPenalty': 84, 'StillRings': 85,
                       'SumoWrestling': 86, 'Surfing': 87, 'Swing': 88, 'TableTennisShot': 89, 'TaiChi': 90,
                       'TennisSwing': 91, 'ThrowDiscus': 92, 'TrampolineJumping': 93, 'Typing': 94, 'UnevenBars': 95,
                       'VolleyballSpiking': 96, 'WalkingWithDog': 97, 'WallPushups': 98, 'WritingOnBoard': 99,
                       'YoYo': 100}
        kinetics400_dict = {'abseiling': 0, 'air_drumming': 1, 'answering_questions': 2, 'applauding': 3,
                            'applying_cream': 4, 'archery': 5, 'arm_wrestling': 6, 'arranging_flowers': 7,
                            'assembling_computer': 8, 'auctioning': 9, 'baby_waking_up': 10, 'baking_cookies': 11,
                            'balloon_blowing': 12, 'bandaging': 13, 'barbequing': 14, 'bartending': 15,
                            'beatboxing': 16, 'bee_keeping': 17, 'belly_dancing': 18, 'bench_pressing': 19,
                            'bending_back': 20, 'bending_metal': 21, 'biking_through_snow': 22, 'blasting_sand': 23,
                            'blowing_glass': 24, 'blowing_leaves': 25, 'blowing_nose': 26, 'blowing_out_candles': 27,
                            'bobsledding': 28, 'bookbinding': 29, 'bouncing_on_trampoline': 30, 'bowling': 31,
                            'braiding_hair': 32, 'breading_or_breadcrumbing': 33, 'breakdancing': 34,
                            'brush_painting': 35, 'brushing_hair': 36, 'brushing_teeth': 37, 'building_cabinet': 38,
                            'building_shed': 39, 'bungee_jumping': 40, 'busking': 41, 'canoeing_or_kayaking': 42,
                            'capoeira': 43, 'carrying_baby': 44, 'cartwheeling': 45, 'carving_pumpkin': 46,
                            'catching_fish': 47, 'catching_or_throwing_baseball': 48,
                            'catching_or_throwing_frisbee': 49, 'catching_or_throwing_softball': 50, 'celebrating': 51,
                            'changing_oil': 52, 'changing_wheel': 53, 'checking_tires': 54, 'cheerleading': 55,
                            'chopping_wood': 56, 'clapping': 57, 'clay_pottery_making': 58, 'clean_and_jerk': 59,
                            'cleaning_floor': 60, 'cleaning_gutters': 61, 'cleaning_pool': 62, 'cleaning_shoes': 63,
                            'cleaning_toilet': 64, 'cleaning_windows': 65, 'climbing_a_rope': 66, 'climbing_ladder': 67,
                            'climbing_tree': 68, 'contact_juggling': 69, 'cooking_chicken': 70, 'cooking_egg': 71,
                            'cooking_on_campfire': 72, 'cooking_sausages': 73, 'counting_money': 74,
                            'country_line_dancing': 75, 'cracking_neck': 76, 'crawling_baby': 77, 'crossing_river': 78,
                            'crying': 79, 'curling_hair': 80, 'cutting_nails': 81, 'cutting_pineapple': 82,
                            'cutting_watermelon': 83, 'dancing_ballet': 84, 'dancing_charleston': 85,
                            'dancing_gangnam_style': 86, 'dancing_macarena': 87, 'deadlifting': 88,
                            'decorating_the_christmas_tree': 89, 'digging': 90, 'dining': 91, 'disc_golfing': 92,
                            'diving_cliff': 93, 'dodgeball': 94, 'doing_aerobics': 95, 'doing_laundry': 96,
                            'doing_nails': 97, 'drawing': 98, 'dribbling_basketball': 99, 'drinking': 100,
                            'drinking_beer': 101, 'drinking_shots': 102, 'driving_car': 103, 'driving_tractor': 104,
                            'drop_kicking': 105, 'drumming_fingers': 106, 'dunking_basketball': 107, 'dying_hair': 108,
                            'eating_burger': 109, 'eating_cake': 110, 'eating_carrots': 111, 'eating_chips': 112,
                            'eating_doughnuts': 113, 'eating_hotdog': 114, 'eating_ice_cream': 115,
                            'eating_spaghetti': 116, 'eating_watermelon': 117, 'egg_hunting': 118,
                            'exercising_arm': 119, 'exercising_with_an_exercise_ball': 120, 'extinguishing_fire': 121,
                            'faceplanting': 122, 'feeding_birds': 123, 'feeding_fish': 124, 'feeding_goats': 125,
                            'filling_eyebrows': 126, 'finger_snapping': 127, 'fixing_hair': 128,
                            'flipping_pancake': 129, 'flying_kite': 130, 'folding_clothes': 131, 'folding_napkins': 132,
                            'folding_paper': 133, 'front_raises': 134, 'frying_vegetables': 135,
                            'garbage_collecting': 136, 'gargling': 137, 'getting_a_haircut': 138,
                            'getting_a_tattoo': 139, 'giving_or_receiving_award': 140, 'golf_chipping': 141,
                            'golf_driving': 142, 'golf_putting': 143, 'grinding_meat': 144, 'grooming_dog': 145,
                            'grooming_horse': 146, 'gymnastics_tumbling': 147, 'hammer_throw': 148, 'headbanging': 149,
                            'headbutting': 150, 'high_jump': 151, 'high_kick': 152, 'hitting_baseball': 153,
                            'hockey_stop': 154, 'holding_snake': 155, 'hopscotch': 156, 'hoverboarding': 157,
                            'hugging': 158, 'hula_hooping': 159, 'hurdling': 160, 'hurling_sport': 161,
                            'ice_climbing': 162, 'ice_fishing': 163, 'ice_skating': 164, 'ironing': 165,
                            'javelin_throw': 166, 'jetskiing': 167, 'jogging': 168, 'juggling_balls': 169,
                            'juggling_fire': 170, 'juggling_soccer_ball': 171, 'jumping_into_pool': 172,
                            'jumpstyle_dancing': 173, 'kicking_field_goal': 174, 'kicking_soccer_ball': 175,
                            'kissing': 176, 'kitesurfing': 177, 'knitting': 178, 'krumping': 179, 'laughing': 180,
                            'laying_bricks': 181, 'long_jump': 182, 'lunge': 183, 'making_a_cake': 184,
                            'making_a_sandwich': 185, 'making_bed': 186, 'making_jewelry': 187, 'making_pizza': 188,
                            'making_snowman': 189, 'making_sushi': 190, 'making_tea': 191, 'marching': 192,
                            'massaging_back': 193, 'massaging_feet': 194, 'massaging_legs': 195,
                            'massaging_persons_head': 196, 'milking_cow': 197, 'mopping_floor': 198,
                            'motorcycling': 199, 'moving_furniture': 200, 'mowing_lawn': 201, 'news_anchoring': 202,
                            'opening_bottle': 203, 'opening_present': 204, 'paragliding': 205, 'parasailing': 206,
                            'parkour': 207, 'passing_American_football_in_game': 208,
                            'passing_American_football_not_in_game': 209, 'peeling_apples': 210,
                            'peeling_potatoes': 211, 'petting_animal_not_cat': 212, 'petting_cat': 213,
                            'picking_fruit': 214, 'planting_trees': 215, 'plastering': 216, 'playing_accordion': 217,
                            'playing_badminton': 218, 'playing_bagpipes': 219, 'playing_basketball': 220,
                            'playing_bass_guitar': 221, 'playing_cards': 222, 'playing_cello': 223,
                            'playing_chess': 224, 'playing_clarinet': 225, 'playing_controller': 226,
                            'playing_cricket': 227, 'playing_cymbals': 228, 'playing_didgeridoo': 229,
                            'playing_drums': 230, 'playing_flute': 231, 'playing_guitar': 232, 'playing_harmonica': 233,
                            'playing_harp': 234, 'playing_ice_hockey': 235, 'playing_keyboard': 236,
                            'playing_kickball': 237, 'playing_monopoly': 238, 'playing_organ': 239,
                            'playing_paintball': 240, 'playing_piano': 241, 'playing_poker': 242,
                            'playing_recorder': 243, 'playing_saxophone': 244, 'playing_squash_or_racquetball': 245,
                            'playing_tennis': 246, 'playing_trombone': 247, 'playing_trumpet': 248,
                            'playing_ukulele': 249, 'playing_violin': 250, 'playing_volleyball': 251,
                            'playing_xylophone': 252, 'pole_vault': 253, 'presenting_weather_forecast': 254,
                            'pull_ups': 255, 'pumping_fist': 256, 'pumping_gas': 257, 'punching_bag': 258,
                            'punching_person_boxing': 259, 'push_up': 260, 'pushing_car': 261, 'pushing_cart': 262,
                            'pushing_wheelchair': 263, 'reading_book': 264, 'reading_newspaper': 265,
                            'recording_music': 266, 'riding_a_bike': 267, 'riding_camel': 268, 'riding_elephant': 269,
                            'riding_mechanical_bull': 270, 'riding_mountain_bike': 271, 'riding_mule': 272,
                            'riding_or_walking_with_horse': 273, 'riding_scooter': 274, 'riding_unicycle': 275,
                            'ripping_paper': 276, 'robot_dancing': 277, 'rock_climbing': 278,
                            'rock_scissors_paper': 279, 'roller_skating': 280, 'running_on_treadmill': 281,
                            'sailing': 282, 'salsa_dancing': 283, 'sanding_floor': 284, 'scrambling_eggs': 285,
                            'scuba_diving': 286, 'setting_table': 287, 'shaking_hands': 288, 'shaking_head': 289,
                            'sharpening_knives': 290, 'sharpening_pencil': 291, 'shaving_head': 292,
                            'shaving_legs': 293, 'shearing_sheep': 294, 'shining_shoes': 295,
                            'shooting_basketball': 296, 'shooting_goal_soccer': 297, 'shot_put': 298,
                            'shoveling_snow': 299, 'shredding_paper': 300, 'shuffling_cards': 301, 'side_kick': 302,
                            'sign_language_interpreting': 303, 'singing': 304, 'situp': 305, 'skateboarding': 306,
                            'ski_jumping': 307, 'skiing_not_slalom_or_crosscountry': 308, 'skiing_crosscountry': 309,
                            'skiing_slalom': 310, 'skipping_rope': 311, 'skydiving': 312, 'slacklining': 313,
                            'slapping': 314, 'sled_dog_racing': 315, 'smoking': 316, 'smoking_hookah': 317,
                            'snatch_weight_lifting': 318, 'sneezing': 319, 'sniffing': 320, 'snorkeling': 321,
                            'snowboarding': 322, 'snowkiting': 323, 'snowmobiling': 324, 'somersaulting': 325,
                            'spinning_poi': 326, 'spray_painting': 327, 'spraying': 328, 'springboard_diving': 329,
                            'squat': 330, 'sticking_tongue_out': 331, 'stomping_grapes': 332, 'stretching_arm': 333,
                            'stretching_leg': 334, 'strumming_guitar': 335, 'surfing_crowd': 336, 'surfing_water': 337,
                            'sweeping_floor': 338, 'swimming_backstroke': 339, 'swimming_breast_stroke': 340,
                            'swimming_butterfly_stroke': 341, 'swing_dancing': 342, 'swinging_legs': 343,
                            'swinging_on_something': 344, 'sword_fighting': 345, 'tai_chi': 346, 'taking_a_shower': 347,
                            'tango_dancing': 348, 'tap_dancing': 349, 'tapping_guitar': 350, 'tapping_pen': 351,
                            'tasting_beer': 352, 'tasting_food': 353, 'testifying': 354, 'texting': 355,
                            'throwing_axe': 356, 'throwing_ball': 357, 'throwing_discus': 358, 'tickling': 359,
                            'tobogganing': 360, 'tossing_coin': 361, 'tossing_salad': 362, 'training_dog': 363,
                            'trapezing': 364, 'trimming_or_shaving_beard': 365, 'trimming_trees': 366,
                            'triple_jump': 367, 'tying_bow_tie': 368, 'tying_knot_not_on_a_tie': 369, 'tying_tie': 370,
                            'unboxing': 371, 'unloading_truck': 372, 'using_computer': 373,
                            'using_remote_controller_not_gaming': 374, 'using_segway': 375, 'vault': 376,
                            'waiting_in_line': 377, 'walking_the_dog': 378, 'washing_dishes': 379, 'washing_feet': 380,
                            'washing_hair': 381, 'washing_hands': 382, 'water_skiing': 383, 'water_sliding': 384,
                            'watering_plants': 385, 'waxing_back': 386, 'waxing_chest': 387, 'waxing_eyebrows': 388,
                            'waxing_legs': 389, 'weaving_basket': 390, 'welding': 391, 'whistling': 392,
                            'windsurfing': 393, 'wrapping_present': 394, 'wrestling': 395, 'writing': 396,
                            'yawning': 397, 'yoga': 398, 'zumba': 399}

        if (self.dataset == "charades"):
            return charades_dict
        elif (self.dataset == 'hmdb51'):
            return hmdb51_dict
        elif (self.dataset == 'ucf101'):
            return ucf101_dict
        elif (self.dataset == "animalkingdom"):
            return animalkingdom_dict
        elif (self.dataset == "kinetics400"):
            return kinetics400_dict

    # Get the data enhancement scheme
    def get_transforms(self, mode='train'):
        self._check()
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        if mode == 'train':
            unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                              GroupRandomHorizontalFlip(True),
                              GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                              GroupRandomGrayscale(p=0.2),
                              GroupGaussianBlur(p=0.0),
                              GroupSolarization(p=0.0)])
            common = Compose([Stack(roll=False),
                              ToTorchFormatTensor(div=True),
                              GroupNormalize(input_mean, input_std)])
            transforms = Compose([unique, common])
            return transforms
        else:
            scale_size = 256

            unique = Compose([GroupScale(scale_size),
                              GroupCenterCrop(input_size)])
            common = Compose([Stack(roll=False),
                              ToTorchFormatTensor(div=True),
                              GroupNormalize(input_mean, input_std)])
            transforms = Compose([unique, common])
            return transforms

    # Data set loader
    def get_data_loader(self, transform, drop_last=False, mode='train'):
        self._check()
        act_dict = self.get_act_dict()
        if (self.dataset == 'charades'):
            data = Charades(self.path, act_dict, total_length=self.total_length, transform=transform,
                            random_shift=False, mode=mode)
        elif (self.dataset == 'animalkingdom'):
            data = AnimalKingdom(self.path, act_dict, total_length=self.total_length, transform=transform,
                                 random_shift=False, mode=mode)
        elif (self.dataset == 'kinetics400'):
            data = kinetics400(self.path, act_dict, total_length=self.total_length, transform=transform,
                               random_shift=False, mode=mode)
        elif (self.dataset == "hmdb51"):
            data = HMDB51(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                          transform=transform, mode=mode)
        elif (self.dataset == "ucf101"):
            data = UCF101(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                          transform=transform, mode=mode)
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        if mode == 'train':
            sampler = RandomSampler(data)
            if self.dataset in ['animalkingdom', 'charades', 'kinetics400']:
                if self.sample != None:
                    sampler = RandomSampler(data, num_samples=self.sample)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler)
            pin_memory = False
        else:
            sampler = DistributedSampler(data) if self.distributed else None
            pin_memory = True
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                            sampler=sampler,pin_memory=pin_memory, drop_last=drop_last)
        return loader

    # 随机获取指定数量的类别
    def get_data_class(self, class_number):
        dict = self.get_act_dict()
        random_keys = random.sample(list(dict.keys()), k=class_number)
        random_values = [dict[key] for key in random_keys]
        return random_keys, random_values

    # 按比例随机分配类别
    def get_split_categories(self,ratio):
        category_dict = self.get_act_dict()
        # 获取所有类别名
        categories = list(category_dict.keys())
        # 随机打乱类别名
        random.shuffle(categories)
        # 计算可见类别的数量
        visible_count = int(len(categories) * ratio)
        # 划分可见类别和不可见类别
        visible_categories = categories[:visible_count]
        invisible_categories = categories[visible_count:]
        # 构建可见类别字典
        visible_dict = {category: category_dict[category] for category in visible_categories}
        # 构建不可见类别字典
        invisible_dict = {category: category_dict[category] for category in invisible_categories}
        return visible_dict, invisible_dict


    # zero
    def get_zero_data_loader(self, transform, drop_last=False, mode='train', label=None):
        self._check()
        act_dict = self.get_act_dict()
        if self.zero is None:
            print("The zero parameter is not set")
            raise
        if self.zero == "EP1":
            if self.dataset == 'hmdb51':
                data = HMDB51(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                              transform=transform, mode=mode, zero=self.zero, label=label)
            if self.dataset == 'ucf101':
                data = UCF101(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                              transform=transform, mode=mode, zero=self.zero, label=label)
        elif self.zero == "EP2":
            if self.dataset == 'hmdb51':
                data = HMDB51(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                              transform=transform, mode=mode, zero=self.zero)
            if self.dataset == 'ucf101':
                data = UCF101(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                              transform=transform, mode=mode, zero=self.zero)
        elif self.zero == "seen":
            if self.dataset == 'hmdb51':
                data = HMDB51(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                              transform=transform, mode=mode, zero=self.zero, label=label)
            if self.dataset == 'charades':
                data = Charades(self.path, act_dict, total_length=self.total_length, transform=transform, mode=mode,
                                zero=self.zero, label=label)
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        if mode == 'train':
            if self.zero == 'seen':
                sampler = RandomSampler(data)
                if self.distributed:
                    sampler = DistributedSamplerWrapper(sampler)
            else:
                sampler = DistributedSampler(data) if self.distributed else None
            pin_memory = False
        else:
            sampler = DistributedSampler(data) if self.distributed else None
            pin_memory = True
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                            sampler=sampler,pin_memory=pin_memory,drop_last=drop_last)
        return loader

    # few
    def get_few_data_loader(self, transform, drop_last=False, mode='train'):
        if self.few is None:
            print("The few parameter is not set")
            raise
        self._check()
        act_dict = self.get_act_dict()
        if self.dataset == 'hmdb51':
            data = HMDB51(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                          transform=transform, mode=mode, few=self.few)
        elif self.dataset == 'ucf101':
            data = UCF101(self.path, act_dict, total_length=self.total_length, file_number=self.file_number,
                          transform=transform, mode=mode, few=self.few)
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        if mode == 'train':
            sampler = RandomSampler(data)
            if self.distributed:
                sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = DistributedSampler(data) if self.distributed else None
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                            sampler=sampler, drop_last=drop_last)
        return loader