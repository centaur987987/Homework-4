# Title: HW 5_ adjusted chicken problem. I am making an effort to complete this homework for extra credit. I have completed hw 4 and am making an effort towards hw 5 with the time I have left before the due date.
# Author: Andrew Centa
# Description: This program will implement the Pec-King Order game and ultimately solve inference questions related to it

#SEE SUPPORTING PDF INSTRUCTION DOCUMENT FOR CONTEXT AND RUBRIC EXPLANATIONS

# HW 5 NEW:
# 1) updated tournaments to be T = 256


#0. IMPORTS
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, replace
import random
import itertools
import msvcrt

# ========
# ENTITIES 
# ========
@dataclass
class Chicken:# create the agent/chicken class

    index: int
    ObservationMemory: jnp.ndarray
    AbilityBelief: jnp.ndarray
    MyAbility: jnp.ndarray
    MyLocation: jnp.ndarray
    LastObservation: jnp.ndarray
    KingBelief: jnp.ndarray
    sack: int
    sack_owner: jnp.ndarray
    transit_entry_tournament: int = -1


    def move(self, env, ent_set, chick, N, best_region):        
        env, ent_set, chick = Events.relocate_region(env, ent_set, chick, N, best_region) # relocate to transit zone
        env, ent_set = Transfer_to_Battle(env, ent_set, chick) # transfer to battle zone
        return env, ent_set
    
    def watch(self, env, ent_set, chick, N, T_i):
        env, ent_set, chick = Events.region_view(env, ent_set, chick, N) # check the region view
        env, ent_set = UpdateObservationMemory(env, ent_set, chick, T_i) # update the observation memory
        return env, ent_set

    def ShareObservations(self, Chick_j_ID, ent_set): # this action shares the chicken's observation history with another designated chicken. 
        chick_index = int(self.index)
        
        # Get the other chicken's memory
        memory_j = ent_set.chickens[Chick_j_ID].ObservationMemory
        
        # stack the 2 information sources
        new_i_memory = jnp.where(self.ObservationMemory == 0, memory_j, self.ObservationMemory)

            # 3. Create a NEW chicken object with the new memory
        # (Using self because self IS ent_set.chickens[chick_index])
        updated_chicken = replace(self, ObservationMemory=new_i_memory)

        # 4. Update the tuple of chickens
        chick_list = list(ent_set.chickens)
        chick_list[chick_index] = updated_chicken
        
        # 5. Replace the chickens collection inside ent_set
        return replace(ent_set, chickens=tuple(chick_list))
    
    def TransferCrowns(self, Chick_j_ID, crowns, ent_set): #Removes NumberofCrowns from chicken's own sack and puts them in the sack of the chicken with the id
        my_crowns = self.sack # crowns of self chicken
        j_crowns = ent_set.chickens[Chick_j_ID].sack # crowns of j chciken

        # update crowns
        my_crowns = my_crowns - crowns
        j_crowns = j_crowns + crowns

        # update chicken components
        updated_chicken = replace(self, sack = my_crowns)
        updated_j = replace(ent_set.chickens[Chick_j_ID], sack = j_crowns)

        # udpate global
        chick_list = list(ent_set.chickens)
        chick_list[self.index] = updated_chicken
        chick_list[Chick_j_ID] = updated_j

        return replace(ent_set, chickens=tuple(chick_list))
    
    def ExtendSackOwnership(self, Chick_j_ID, ent_set):
        # convenient indices
        i = self.index
        j = Chick_j_ID

        # union of ownership (order-independent, no duplicates)
        shared_owners = list(set(ent_set.chickens[i].sack_owner) | set(ent_set.chickens[j].sack_owner))

        # replace the owner with shared owners
        updated_i = replace(ent_set.chickens[i], sack_owner = shared_owners)
        updated_j = replace(ent_set.chickens[j], sack_owner = shared_owners)

        # updated globally
        chick_list = list(ent_set.chickens)
        chick_list[i] = updated_i
        chick_list[j] = updated_j

        return replace(ent_set, chickens=tuple(chick_list))

@dataclass
class Cage:
    index: int
    MyLocation: jnp.ndarray
    MyInterior: jnp.ndarray
    pass

  #### BRAD TO DO THIS
    #____________________

@dataclass
class Monitor:
    MyLocation: jnp.ndarray
    MyView: jnp.ndarray
    MyAgentSet: jnp.ndarray
    pass

@dataclass
class EntitySet:
    chickens: tuple
    cages: tuple
    monitors: tuple

    @classmethod
    def spawn_entity_set(cls, M, N, T, ability):

        # fill in the chicken spawning data
        chick_list = []
        for i in range(M):
            chicken_instance = Chicken(
                index = i, # index is 1 through M
                ObservationMemory = jnp.zeros((M, M, T)), 
                AbilityBelief = jnp.ones((M, N, N)) / N, # establish the priors
                MyAbility = ability[i],                
                MyLocation = jnp.zeros((3,3)), # matches 1 aspect of the Zone_Array to capture zone and coop
                LastObservation = jnp.zeros((M, M)),
                KingBelief = jnp.ones((M, N, 2)) * jnp.array([15/16, 1/16]),
                sack = 0,
                sack_owner = [i],
            )
            chick_list.append(chicken_instance)

        # fill in the cage spawning data
        cage_list = []
        for i in range(N * 8):
            # hard code in cage locations using boolean values
            coop = jnp.array([i // 8], dtype=jnp.float32)  

            cage_instance = Cage(
                index = i,
                MyLocation = coop,
                MyInterior = jnp.array([-1, -1]) 
            )

            cage_list.append(cage_instance)
        
        # fill in the monitor spawning data
        monitor_list = []
        for i in range(N):
            
            
            monitor_instance = Monitor(
                MyLocation = i, # 1 monitor per region
                MyView = jnp.zeros((N,)),      
                MyAgentSet = jnp.array([-1])  
            )
            monitor_list.append(monitor_instance)

        return cls(
            chickens = tuple(chick_list),
            cages = tuple(cage_list),
            monitors = tuple(monitor_list)
        )

# ============
# ENVIRONMENT 
# ============
@dataclass
class Domain: # setup a data class for the domain    
    ZoneArray: jnp.ndarray 
    ActionArray: jnp.ndarray 
    
    BattleOutcomeMatrix: jnp.ndarray
    BattleHistoryMatrix: jnp.ndarray
    
    # create tensors of zeros to create each element of the environment
    @classmethod
    def spawn_domain(cls, M = 64, N = 4, T = 256):
        Zone_Array = jnp.zeros((M, N, 3))
        Action_Array = jnp.zeros((M, 1))
        Battle_Outcome_Matrix = jnp.zeros((M, M, N))
        Battle_History_Matrix = jnp.zeros((M, M, N, T))

        return cls(Zone_Array, Action_Array, Battle_Outcome_Matrix, Battle_History_Matrix)    
     
# ======
# EVENTS 
# ======
class Events: # Events can then be expressed as functions of the form:  InteractionEvent(Tuple(Environment,EntitySet))-> Tuple(Environment,EntitySet)
              # Translation: events should take current state of the environment and entity set as an input and output a new state

    @staticmethod
    def assign_agents_to_cage(env, ent_set, N, T_i, k): # if chicks are in battle zone, assign to cages
        # initialize a dictionary of Battle ready chickesn in region 1, 2, 3, or 4      
        BattleChicks = {} #iniitlize a battle chick dictionary 
        for i in range(N): # Create battlechick dictionary {0: [], 1: [], 2: [], 3: []}
            BattleChicks[i] = []

        # loop through all chickens to check if their location is battle zone. Create specific lists for chickens in each battle zone
        for chick in ent_set.chickens:
            
            ChickIndex = int(chick.index)
            for coop in range(N):
                # Append battlechick lists by coop
                if jnp.any(chick.MyLocation[coop, 0] == 1.0):
                    BattleChicks[coop].append(ChickIndex) # append the BattleReadyChicks list
            
            # Check if a chicken has lost 4 battles in this region and remove them fromthe BattleCHicks list if so
                LossRecord_k = jnp.sum(env.BattleOutcomeMatrix[ChickIndex, :, :] == -1) # how many battles have each chicken lost out of last 4
                # if loss record = 4 and chicken is in the battle chicks list...
                if (LossRecord_k == 4.0 and ChickIndex in BattleChicks[coop]):
                    BattleChicks[coop].remove(ChickIndex) # remove from battle list for that coop
                    
        new_cage_list = list(ent_set.cages) # initialize a new cage list
        cages_per_region = 8

        for index in range(cages_per_region):
        # cycle through all M/2 cages    
            for coop in range(N):                       
                cage_index = coop * cages_per_region + index # region variable 
                cage = ent_set.cages[cage_index]
                
                current_interior = jnp.array([-1, -1]) # set interior to default to empty each loop                    

                if len(BattleChicks[coop]) >= 2: # move forward as long as 2 or more chicks are waiting to battle
                    found_pair = False # flag for if a pair is found
                    for champion in BattleChicks[coop][:]: # cycle through all champions (left chick)
                        if found_pair: # if a pair is already found, break the cycle
                            break
                         
                        for challenger in BattleChicks[coop][:]: # cycle through all challengers (right chicks)                
                            if champion == challenger: # skip when attacker and challenger are the same. 0 can't fight 0
                                continue

                            has_battled_before = jnp.any(env.BattleOutcomeMatrix[champion, challenger, coop] != 0) # check if they have battled
                            
                            if not has_battled_before: # if they have not battled...                           
                        
                                current_interior = current_interior.at[0].set(champion) # set interior postion 0 be the next regional battle chicken index
                                current_interior = current_interior.at[1].set(challenger)# set interior postion 1 be the next regional battle chicken index
                                
                                BattleChicks[coop].remove(champion) # remove chicken L from battle list
                                BattleChicks[coop].remove(challenger) # remove chicken R from battle list
                                found_pair = True # set found pair to true
                                break
                            
                            if len(BattleChicks[coop]) >= 2 and not found_pair:
                                current_interior = jnp.array([-1, -1])
                                
                    

                new_cage = replace(cage, MyInterior = current_interior) # for this loop, create a new cage that replaces the current interior 
                new_cage_list[cage_index] = new_cage # append the new_cage to the new_cage list
    
        updated_ent_set = replace(ent_set, cages = tuple(new_cage_list)) # create the new updated_ent_set'''      
        for Cage in ent_set.cages:
            #print(f"Cage {Cage.index} (Region {Cage.MyLocation}) assigned: {Cage.MyInterior}") # print statement
            break
        #print(' ')
        return env, updated_ent_set
    
    @staticmethod
    def dominance_battle(env, ent_set): # DONE MINUS DEBUGGING
        current_BattleOutcomeMatrix = env.BattleOutcomeMatrix        
        for cage in ent_set.cages:            
            # skip empty cages
            if int(cage.MyInterior[0]) == -1 or int(cage.MyInterior[1]) == -1:
                continue
            
            Chick_L_Index = int(cage.MyInterior[0]) # Find the index of the chicken in the left side of the cage
            Chick_R_Index = int(cage.MyInterior[1]) # find the inde of the chicken in the right side of the cage
            cage_region = int(cage.MyLocation[0]) # convenient variable to capture the cage region
            
            # inputs for P(chicken in left corner wins|xL,xR)
            x_L = ent_set.chickens[Chick_L_Index].MyAbility[cage_region] # left chickens input parameter
            x_R = ent_set.chickens[Chick_R_Index].MyAbility[cage_region] # right chicken input parameter
        
            # if the ability of L is bigger than R, set win for L_chick and loss for R_chick
            if x_R < x_L:
                current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_L_Index, Chick_R_Index, cage_region].set(1.0)
                current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_R_Index, Chick_L_Index, cage_region].set(-1.0)

            # if ability of R is bigger than L, set win for R and loss for L 
            elif x_R > x_L:
                current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_L_Index, Chick_R_Index, cage_region].set(-1.0)
                current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_R_Index, Chick_L_Index, cage_region].set(1.0)          
            
            # if the abilities are equal, give a random chance that one or the other wins
            elif x_R == x_L:
                random_number = random.randint(1, 2)
                if random_number == 1:
                    current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_L_Index, Chick_R_Index, cage_region].set(1.0)
                    current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_R_Index, Chick_L_Index, cage_region].set(-1.0)       
                elif random_number == 2:
                    current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_L_Index, Chick_R_Index, cage_region].set(-1.0)
                    current_BattleOutcomeMatrix = current_BattleOutcomeMatrix.at[Chick_R_Index, Chick_L_Index, cage_region].set(1.0)
            
            else:
                print('ERROR: No chicken won the dominance battle') 
        
        updated_env = replace(env, BattleOutcomeMatrix = current_BattleOutcomeMatrix)                
        return updated_env, ent_set
    
    @staticmethod 
    def region_view(env, ent_set, chick, N): # DONE - but "watch" region chose randomly
        # Chickens must be in a BattleZone or SpectatorZone to initiate.  All chickens in a coop's Battlezone receive only that coop's 
        # RegionView automatically. All SpectatorZone chickens that choose the RegionView action receive the a view of the coop they select.
        current_LastObservation = chick.LastObservation
        chick_region = int(jnp.argmax(jnp.sum(chick.MyLocation, axis=1))) # get chick region
        
        all_region = np.arange(0, N) # all regions in the world
        possible_regions = all_region[all_region != chick_region] # all regions except the chicks current region        
        rand_region = np.random.choice(possible_regions) # choose a random region to go to

        just_fought = jnp.any(env.BattleOutcomeMatrix[chick.index, :, :] != 0)
        
        if just_fought: # if a chicken just fought
            current_LastObservation = env.BattleOutcomeMatrix[:, :, chick_region] # current last observation is of its own region
        else:
            current_LastObservation = env.BattleOutcomeMatrix[:, :, rand_region] # current last observation is of a region of its choice

        updated_chick = replace(chick, LastObservation=current_LastObservation) # replace previous chick last observation with new observation
        new_chick_list = list(ent_set.chickens) # create list of all chickesn
        new_chick_list[chick.index] = updated_chick # make the new chick list for the chick index in question match the new chick
        updated_ent_set = replace(ent_set, chickens=tuple(new_chick_list)) # update and raplace the chicken in the entire ent_set
        return env, updated_ent_set, updated_chick
    
    @staticmethod
    def relocate_zone(env, ent_set): # DONE MINUS DEBUGGING
        #Chickens in the BattleZone that fail to be paired in the AssignAgentsToCage event trigger this 
        # event which moves them to the SpectatorZone. Pairing fails if a chicken has fought all the other 
        # chickens, has lost k or more rounds, or is the last to be paired for an odd number of chicken (pairing 
        # only works for an even number of chickens). 
        current_ZoneArray = env.ZoneArray
        cage_interior_list = []

        # List all chickens currently in cages
        for cage in ent_set.cages:
            cage_interior_Left = cage.MyInterior[0] # chicken in left of the cage
            cage_interior_Right = cage.MyInterior[1] # chicken in right of the cage
 
            cage_interior_list.append(cage_interior_Left) # append the chicken list with the left chicken
            cage_interior_list.append(cage_interior_Right) # append the chicken list with the left chicken
        
        # loop through all chickens to check if their location is battle zone or spetator zone.
        for chick in ent_set.chickens:
            chick_region = int(jnp.argmax(jnp.sum(chick.MyLocation, axis=1))) # which region is the chicken in
            if (jnp.any(chick.MyLocation[:, 0] == 1.0) and  # if a chicken in is the battle zone
                chick.index not in cage_interior_list): # AND a chicken is not in a cage

                current_ZoneArray = current_ZoneArray.at[chick.index, chick_region, 0].set(0) # set the Zone Array battle zone region location to 0 for that chicken
                current_ZoneArray = current_ZoneArray.at[chick.index, chick_region, 1].set(1) # set the Zone Array spectator zone region location to 1 for that chicken
                
            else:
                #print("relocate zone has failed for chicken:", chick.index)
                pass

        updated_env = replace(env, ZoneArray = current_ZoneArray) # update the environment
        updated_env, updated_ent_set = UpdateChickLocation(updated_env, ent_set) # update the chicken location as well
        return updated_env, updated_ent_set
    
    @staticmethod 
    def relocate_region(env, ent_set, chick, N, best_region): # DONE-but region move happens randomly            
        current_ZoneArray = env.ZoneArray # create current zone array
        chick_region = int(jnp.argmax(jnp.sum(chick.MyLocation, axis=1))) # chicks current region
        
        all_region = np.arange(0, N) # all regions in the world        
        possible_regions = all_region[all_region != chick_region] # all regions except the chicks current region        
        rand_region = np.random.choice(possible_regions) # choose a random region to go to

        # move the chicken
        if best_region == False:
            rand_region = rand_region
        elif best_region != False:
            rand_region = best_region

        
        current_ZoneArray = current_ZoneArray.at[chick.index, :, :].set(0) # set all location info to be 0
        current_ZoneArray = current_ZoneArray.at[chick.index, rand_region, 2].set(1) # set transfer zone to be 1
        # update environment
        updated_env = replace(env, ZoneArray = current_ZoneArray) # update the environment

        updated_env, updated_ent_set = UpdateChickLocation(updated_env, ent_set) # update the chicken location as well
        updated_chick = updated_ent_set.chickens[chick.index]
        return updated_env, updated_ent_set, updated_chick

    @staticmethod    
    def init(M, N, T): # function to initialize domain
        env = Domain.spawn_domain(M , N , T) # spawn the environment
        ability = Spawn_Ability_Score(M, N) # spawn the ability matrix
        ent_set = EntitySet.spawn_entity_set(M, N, T, ability) # spawn the entities
        env, ent_set = Chick2BattleZone(env, ent_set, M, N) # get chicks ready to battle in the battle zone

        return env, ent_set

    @staticmethod
    def new_tournament(env, ent_set, M, N): # initial print outs
        print("\nBattle zone count:", int(jnp.sum(env.ZoneArray[:, :, 0])))           
           
        return env, ent_set
    
    @staticmethod
    def tournament(env, ent_set, T_i, M, N, k): # Typical tournament loop 
        print("Tournament:" ,T_i)
        #Battle = 0
        while True: # continue until all cages are 
            #print("Battle:", Battle)
            #Battle += 1
            env, ent_set = Events.assign_agents_to_cage(env, ent_set, N, T_i, k)
            env, ent_set = Events.relocate_zone(env, ent_set)
            env, ent_set = UpdateChickLocation(env, ent_set)
            env, ent_set = Events.dominance_battle(env, ent_set)   
            
            

            Cage_Interiors = jnp.stack([cage.MyInterior for cage in ent_set.cages]) # make an array of all cage interiors
            if jnp.all(Cage_Interiors == -1): # break the loop if all cages are empty
                break 
        
        for chick in ent_set.chickens:
            if jnp.any(env.BattleOutcomeMatrix[chick.index, :, :] != 0):  # only chickens that fought
                env, ent_set, _ = Events.region_view(env, ent_set, ent_set.chickens[chick.index], N)
                env, ent_set = UpdateObservationMemory(env, ent_set, ent_set.chickens[chick.index], T_i)
       
        env, ent_set = UpdateAbilityBelief(env, ent_set, N) 
        return env, ent_set
    
    @staticmethod
    def close_tournament(env, ent_set, T_i, M, N, k): # reserved function 
        #print('Battle Complete')
        
        # Record Kings
        World_King_List = [] # initialize list
        for coop in range(N): # cycle thrgouh coops
            regional_king_list = Get_King(env, ent_set, coop) # get kings for each coop
            World_King_List.append(regional_king_list) # append the kings list

        # flatten king list
        flat_World_King_List = [] # initialize flattened list
        for coop in World_King_List: # cycle through king list
            for tuple in coop: # cycle through coops
                flat_World_King_List.append(tuple) # append the flattened list

        #print(flat_World_King_List)
        # Chicken Action Policy
        # kings want to stay in their coop
        king_ids = {tup[0] for tup in flat_World_King_List} # create array for all king ids
        env, ent_set = ChickActionPolicy2(env, ent_set, king_ids, N, T_i, M)

        # Continue tournament looop
        env, ent_set = UpdateKingBelief(env, ent_set, flat_World_King_List)
        env, ent_set = UpdateBattleHistoryMatrix(env, ent_set, T_i)
        env, ent_set = Reset_BattleOutcomeMatrix(env, ent_set, M, N) # reset the battle outcome matrix

        return env, ent_set, flat_World_King_List
        
# ===============
# EXTRA FUNCTIONS 
# ===============
def Chick2BattleZone(env, ent_set, M, N):# send all chickens to the battle zone of their coops. See "Zone Array" Description at the top

    chicks_per_coop = M // N  # 16 chickens per coop

    chick_indices = jnp.arange(M)                       
    coop_assignments = chick_indices // chicks_per_coop    

    # One vectorized write: set battle zone (col 0) to 1 for all chickens at once
    updated_ZoneArray = env.ZoneArray.at[chick_indices, coop_assignments, 0].set(1.0)
    updated_env = replace(env, ZoneArray=updated_ZoneArray)
    updated_env, updated_ent_set = UpdateChickLocation(updated_env, ent_set)

    return updated_env, updated_ent_set

def Get_King(env, ent_set, coop): # print the kings after a battle is complete
                
        loss_list = [] # initialize wins list
        king_list = [] # initialize kings list (if more than 1)       
        
        for chick in ent_set.chickens:
            chick_region = jnp.argmax((chick.MyLocation == 1).any(axis=1)) # region chicken is in
            
            has_battled = (env.BattleOutcomeMatrix[chick.index] != 0).any()

            if chick_region == coop and has_battled: # if chicken is in region of this Get_King function...
                losses = jnp.sum(env.BattleOutcomeMatrix[chick.index] == -1.0) # Record how many times each chicken lost

                #print("chicken", chick.index, "earned a losing score of ", losses, "in region", chick_region )
                loss_list.append((chick.index, int(losses))) # create a tuple of losing chicken and losses
        
        if not loss_list:
            return [] # Return empty if nobody in this coop fought
    
        Coops_Best = min(loss_list, key=lambda x: x[1])
        # for chick tuples in the loss list...    
        for chick_tuple in loss_list:
            chick_id = chick_tuple[0] # chick ID is first vlaue
            loss_count = chick_tuple[1] # score is second value
            
            if loss_count == 0.0: # if score is 0...
                king_list.append((chick_id, coop, loss_count)) # add them to the king list
            
        if king_list == []:        
            king_list.append((Coops_Best[0], coop, loss_count))
            pass

        return king_list

def UpdateChickLocation(env, ent_set):# update the chick location
    new_chick_list = [] # initialize a new list of chicken entities
    
    # loop through all chicken entities
    for chick in ent_set.chickens:
        updated_chick = replace(chick, MyLocation = env.ZoneArray[chick.index]) # replace chicken location reference with zone array
        new_chick_list.append(updated_chick) # append the new empty list
    updated_ent_set = replace(ent_set, chickens=tuple(new_chick_list)) # replace the ent set with a tuple of the new list

    return env, updated_ent_set
          
def UpdateBattleHistoryMatrix(env, ent_set, T_i):# Add the newest Battle Outcome Matrix to the Battle History Matrix  
    current_BattleHistoryMatrix = env.BattleHistoryMatrix   # set a current Battle History Matrix      
    current_BattleOutcomeMatrix = env.BattleOutcomeMatrix   # set a current Battle Outcome matrix
    
    # set the 4th dimension of the battle history matrix to equal the current rendition of the battle outcome matrix
    current_BattleHistoryMatrix = current_BattleHistoryMatrix.at[: ,: , : ,T_i].set(current_BattleOutcomeMatrix)
    updated_env = replace(env, BattleHistoryMatrix = current_BattleHistoryMatrix) # update the environment
    return updated_env, ent_set

def Spawn_Ability_Score(M, N): # Create abilities to attach to each chicken
    # poisson distribution for deciding probabilities
    ability_numbers = np.arange(1, 5)
    numerator = np.exp(-ability_numbers / (N / 3)) # numerator from homework
    denominator = sum(numerator) # denominator from homework
    ability_prob = numerator / denominator # normalise so probabilities sum to 1
    
    # apply the poisson distribution to [1,2,3,4] but use seeds for debugging
    seed = 42 # generate a random, but standardized setup for debuggin
    ability_numbers = list(range(1, 5))  # the ability numbers can be 1 through 4
    seed_random = np.random.default_rng(seed)   # randomly generate abilities via a seed for sharing results
    
    # create the scores
    scores = seed_random.choice(ability_numbers, size = (M, N),  replace=True, p = ability_prob)
    scores_array = jnp.array(scores, dtype = jnp.int32) # convert to Jax

    return scores_array

def Reset_BattleOutcomeMatrix(env, ent_set, M, N):# Reset BattleOutcomeMatrix back to zeroes for next tournament
    current_BattleOutcomeMatrix = env.BattleOutcomeMatrix # define the current Battle Outcome Matrix
    current_BattleOutcomeMatrix = jnp.zeros((M, M, N)) # reset back to zeroes

    updated_env = replace(env, BattleOutcomeMatrix = current_BattleOutcomeMatrix) # update the environment
    return updated_env, ent_set

def Transfer_to_Battle(env, ent_set, chick): # move chickens from the transit zone to that battle zone
    current_ZoneArray = env.ZoneArray # current array
 
    chick_region = int(jnp.argmax(jnp.sum(chick.MyLocation, axis=1))) # find chicke region
      
    if jnp.any(current_ZoneArray[chick.index, :, 1:3] == 1.0): # check for 1 in specator or transfer zone
        current_ZoneArray = current_ZoneArray.at[chick.index, :, :].set(0)
        current_ZoneArray = current_ZoneArray.at[chick.index, chick_region, 0].set(1)

    updated_env = replace(env, ZoneArray=current_ZoneArray) # update zone array
    updated_env, updated_ent_set = UpdateChickLocation(updated_env, ent_set) # update chicken location
    
    return updated_env, updated_ent_set

def UpdateObservationMemory(env, ent_set, chick, T_i): # update the observation history
    current_ObservationMemory = chick.ObservationMemory   # set a current ObservationMemory    
    current_LastObservation = chick.LastObservation   # set a current LastObservation

    # set the next iteration of Observation memory to be the last observation
    current_ObservationMemory = current_ObservationMemory.at[:, :, T_i].set(current_LastObservation)

    # set the 3rd dimension of the ObservationMemory to equal the current rendition of the LastObservation
    updated_chick = replace(chick, ObservationMemory=current_ObservationMemory)
    new_chick_list = list(ent_set.chickens)
    new_chick_list[chick.index] = updated_chick
    updated_ent_set = replace(ent_set, chickens=tuple(new_chick_list))

    return env, updated_ent_set

def UpdateAbilityBelief(env, ent_set, N): # adjust the probability of each chickens score in each zone
    updated_chick_list = list(ent_set.chickens) # list all chickens in a new structure
    
    for chick in updated_chick_list: # cycle through all chickens
        chick_ID = chick.index # chick index
        updated_belief = chick.AbilityBelief
        
        for competitor in ent_set.chickens:# cycle through all competitors
            #input("Press Enter to continue...")
            comp_ID = competitor.index # competitor index
            fought = jnp.any(env.BattleOutcomeMatrix[comp_ID] != 0) # check to see if competitor has fought 
            observed = jnp.any(ent_set.chickens[chick_ID].ObservationMemory[comp_ID] != 0) # chekc to see if competiro was observed
            #print(ent_set.chickens[chick_ID].ObservationMemory[comp_ID])
            #print(observed)
            #if not fought or not observed: # if not, skip
            if not fought:
                continue
            comp_region = int(jnp.argmax(jnp.any(env.BattleOutcomeMatrix[comp_ID] != 0, axis=0))) # region of the competitor from battle matrix
            
            if chick_ID == comp_ID: # ignore when the idices are the same
                # poppulate the chickens known ability for itself
                for coop in range(N): # loop through the coops
                    ability = int(chick.MyAbility[coop]) # get the chickens ability for this coop
                    row = jnp.zeros(N).at[ability - 1].set(1.0) # make an array of 0's with 1, 1
                    updated_belief = updated_belief.at[chick_ID, coop, :].set(row) # set the proper line to be that line
                continue
            
            observation = chick.LastObservation
            
            if observation[comp_ID, chick_ID] == 1.0: # if competitior won
                prob =  jnp.array([.1, .2, .3, .4])
            elif observation[comp_ID, chick_ID] == -1.0: # if competitor lost
                prob = jnp.array([.4, .3, .2, .1])
            else:
                continue

            updated_row = updated_belief[comp_ID, comp_region, :] * prob # find product of prior and prob
            updated_row = updated_row / jnp.sum(updated_row) #normalize rows
            updated_belief = updated_belief.at[comp_ID, comp_region, :].set(updated_row) # set each row to normalized
        
        updated_chick_list[chick_ID] = replace(chick, AbilityBelief = updated_belief) # update chick list
    
    updated_ent_set = replace(ent_set, chickens = tuple(updated_chick_list)) # updated ent_set

    return env, updated_ent_set

def UpdateKingBelief(env, ent_set, King_List): # Update the king belief tensor
    updated_chick_list = list(ent_set.chickens) # list all chickens in a new structure
    
    for chick in updated_chick_list: # cycle through all chickens
        chick_ID = chick.index # chick index
        updated_belief = chick.KingBelief
        
        for competitor in ent_set.chickens:# cycle through all competitors
            comp_ID = competitor.index # competitor index
            fought = jnp.any(env.BattleOutcomeMatrix[comp_ID] != 0) # check to see if competitor has fought 
            if not fought: # if not, skip
                continue
            comp_region = int(jnp.argmax(jnp.any(env.BattleOutcomeMatrix[comp_ID] != 0, axis=0))) # region of the competitor from battle matrix
            
            if chick_ID == comp_ID: # ignore when the idices are the same
                # poppulate the chickens known ability for itself
                continue
            observation = chick.LastObservation
            king_array = jnp.array(King_List)
            np.set_printoptions(threshold=np.inf)

            
            if comp_ID in king_array[:, 0] and jnp.any(observation[:, comp_ID]) != 0.0: # if competitior won and chick saw it
                prob =  jnp.array([.05, .95])
            elif comp_ID not in king_array[:, 0] and jnp.any(observation[:, comp_ID]) != 0.0:  # if competitor lost and chick saw it
                prob = jnp.array([.95, .05])
            else:
                continue

            updated_row = updated_belief[comp_ID, comp_region, :] * prob # find product of prior and prob
            updated_row = updated_row / jnp.sum(updated_row) #normalize rows
            updated_belief = updated_belief.at[comp_ID, comp_region, :].set(updated_row) # set each row to normalized
            
        updated_chick_list[chick_ID] = replace(chick, KingBelief = updated_belief) # update chick list
    
    updated_ent_set = replace(ent_set, chickens = tuple(updated_chick_list)) # updated ent_set
    np.set_printoptions(suppress=True, precision=4)
    
    return env, updated_ent_set   

def CountCrowns(king_list, M, ent_set): # get an empirical count of how many crowns each chicken has
    emp_king_list = np.zeros((M, 2)) # create an array of M,2 shape
    
    for chick_ID in range(M): # loop through each chick_ID
        emp_king_list[chick_ID, 0] = chick_ID #Assign column 0 of array to be ID
        king_count = sum(1 for tournament in king_list for crown in tournament if crown[0] == chick_ID) # find how many crowns each chicken has
        emp_king_list[chick_ID, 1] = king_count # assign column 1 to be crowns

        ent_set.chickens[chick_ID].sack = king_count

    #print(' ')
    #print('[(chick) (# of crowns)]')
    #print(emp_king_list)
    return ent_set, emp_king_list

def ShareCount(ent_set, emp_king_list): # if chickens are sharing crowns, develop the adjusted emp_king_list with the shared chickens
    for i in emp_king_list: # check the current king list 
        total_crowns = 0 # initialize total crowns
        index = int(i[0]) # variable for index
        sack_owners = ent_set.chickens[index].sack_owner # variable for sack owner
        num_owners = len(sack_owners) # track number of ownesrs
        if num_owners > 1: # if the number of owners is greater than 1
            for owner in sack_owners: # loop through owners
                owner_crowns = ent_set.chickens[owner].sack # reference how many crowns each owner has
                total_crowns += owner_crowns # add the total crowns

            # ensure there are no decimals
            base_share = total_crowns // num_owners   # integer division
            remainder = total_crowns % num_owners     # leftover crowns

            # give remainder to THIS chicken (the one being evaluated)
            if index in sack_owners:
                total_crowns = base_share + remainder
            else:
                total_crowns = base_share
            emp_king_list[index, 1] = total_crowns # update empiracl king list
            

        elif num_owners == 1: # if owners is 1 long
            total_crowns = ent_set.chickens[index].sack # total crowns equals that owners total
            emp_king_list[index, 1] = total_crowns

    return emp_king_list

def ChickActionPolicy2(env, ent_set, king_ids, N, T_i, M): # policy used to ensure a converging result
    for chick in ent_set.chickens: # loop through all chickens to find actions
        best_region = False # initialize best region as false if using this policy
        if chick.index in king_ids:
            env, ent_set = Transfer_to_Battle(env, ent_set, chick)
        elif jnp.any(chick.MyLocation[:, 1] == 1.0): # if a chicken in is the spectator zone
            coin_toss = random.randint(1, 2) # randomly move or switch for the moment
            if coin_toss == 1:
                env, ent_set = chick.move(env, ent_set, chick, N, best_region)
            if coin_toss == 2: 
                env,ent_set = chick.watch(env, ent_set, chick, N, T_i)
    return env, ent_set
                
def ChickActionPolicy(env, ent_set, king_ids, N, T_i, M): # experimental chicken policy that uses beliefs
    for chick in ent_set.chickens: # loop through all chickens to find actions

        # If a chicken was a king in the last fight, stay in that coop
        if chick.index in king_ids:
            env, ent_set = Transfer_to_Battle(env, ent_set, chick)
        
        # if a chicken in is the spectator zone
        elif jnp.any(chick.MyLocation[:, 1] == 1.0): 

            region_scores = [] # initialize a region score array

            # for each region, get a region score from PolicyCHeck()
            for region in range(N):
                averaged_score =  Policy_Check(chick, region, M, N) # get the score
                region_scores.append(averaged_score) # apend the score list
            region_best = int(np.argmax(region_scores)) # find the likely best region
            
            # if region score is low, go there for a fight
            recent_history = env.BattleHistoryMatrix[chick.index, :, :, T_i-4:T_i] # check last 4 rounds to see if chicken faught
            has_fought = jnp.sum(recent_history != 0) # has the chicken faught
            if has_fought == 0 and T_i > 5:
                 env, ent_set = chick.move(env, ent_set, chick, N, np.random.randint(0, N))
            else:
                if region_scores[region_best] > 0.45:
                    env, ent_set = chick.move(env, ent_set, chick, N, region_best)
                
                # if the region score is low, avoid fighting
                else:
                    env, ent_set = chick.watch(env, ent_set, chick, N, T_i)

    return env, ent_set

def Policy_Check(chick, region, M, N): # develop a score to enter into the action policy


    my_ability = int(chick.MyAbility[region]) # reference chicks own ability for this region
    belief = chick.AbilityBelief[:, region, :]  # reference chickens belief of different regions
    belief = np.delete(belief, chick.index, axis=0) # don't factor for the chickens self
    p_weak = np.sum(belief[:, :my_ability - 1], axis=1) # find the probablity that other opponents are weaker
    score = np.mean(p_weak) # use mean to create a score by finding the average of p_win_all
    
    # factor in for kings
    king_probs = chick.KingBelief[:, region, 1]
    avg_king_prob = np.mean(king_probs)

    return score * (1 - 0.5 * avg_king_prob) # since the king value is likely so small, take its inverse probabilty to ensure the number doesn't have too many decimals.

def Pivsj(chicken, competitor, region, ent_set, N): # find probabilty i best j
    # find the ability belief of each chicken given a region
    ability_i = ent_set.chickens[chicken].MyAbility[region] # known i ability
    i_belief_of_j = ent_set.chickens[chicken].AbilityBelief[competitor, region, :] # chicken i observation of j ability

    prob = 0.0 # initialize probabilty
    
    # index through ability indices
    
    for ability_j in range(N): # loop through all j values
        p_aj = i_belief_of_j[ability_j]
        
        if ability_i > ability_j: # if ability i is better than j, i always wins
            win_prob = 1.0
        elif ability_i < ability_j: # if i is less than j, j always wins
            win_prob = 0.0
        else:
            win_prob = 0.5 # if the abilities are equal, they tie
        
        prob += win_prob * p_aj
            
    return prob

def PivsjCleanPrint(chicken_i_ID, ent_set, N, M): # created a clean print out based on feedback from TA: Felix
    chicken_i = ent_set.chickens[chicken_i_ID] 

    # Print Title Block
    print("=" * 80)
    print(f"P(chicken {chicken_i_ID} beats j | obs history)   [observer={chicken_i_ID}]")
    print("Each row = one opponent j, columns = coops")
    print("=" * 80)

    # Print Header
    header = f"{'j':<5}"
    for coop in range(N):
        header += f"Coop{coop:<5}"
    header += "True abilities (i vs j)"
    print(header)

    # Rows
    for j in range(M):
        if j == chicken_i_ID: # skip chicken if it is itself
            continue

        pivsj_PrintOut = f"{j:<5}"

        # print data
        probs = []
        for coop in range(N):
            p = Pivsj(chicken_i_ID, j, coop, ent_set, N) # run Pivsj
            probs.append(p)
            pivsj_PrintOut += f"{p:<9.3f}"

        # print chicken abilities
        i_abilities = [int(x) for x in chicken_i.MyAbility]
        j_abilities = [int(x) for x in ent_set.chickens[j].MyAbility]
        pivsj_PrintOut += f"i={i_abilities} j={j_abilities}"


        # Print Samples for Chicken 0
        print(pivsj_PrintOut)

def P_i_king(ent_set, chick_ID, region, N, M): # find probabilty of any chicken being king

    PiKing = 1.0 # initialize
    
    for competitor_ID in range(M): #loop accross competitors            
        if competitor_ID == chick_ID:
            continue
    
        PiKing *= Pivsj(chick_ID, competitor_ID, region, ent_set, N) # find the product of all Pivsj for a coop and specific chicken
    return PiKing
            
def P_i_king_Print(ent_set, N, M): # print P i king
    
    # Print Title Block``
    print("=" * 80)
    print(f"P(chicken i king | obs history)")
    print("Each row = one chicken i, columns = coops")
    print("=" * 80)

    # Print Header
    header = f"{'i':<5}"
    for coop in range(N):
        header += f"Coop{coop:<5}"
    print(header)

    # Rows
    for i in range(M):
        pking_PrintOut = f"{i:<5}"

        # print data
        for coop in range(N):
            p = P_i_king(ent_set, i, coop, N, M) # run Pivsj
            pking_PrintOut += f"{p:<9.3f}"

        # Print Samples for Chicken 0
        print(pking_PrintOut)

def P_ability_profile(chick_ID, competitor_ID, ent_set, N): # fidn the probabily of the abilty profile
    ab_grid = ent_set.chickens[chick_ID].AbilityBelief[competitor_ID]
    #print(ab_grid)
    scores = [1, 2, 3, 4] # possible scores
    all_combos = list(itertools.product(scores, repeat=4))

    all_combos_list = [list(combo) for combo in all_combos] # format into list of lists

    prob_combo_list = [] # initialize the list
    
    for combo in all_combos_list: # loop through all the possible combos and calculate the probability of having that particular score based on the ability belief
        val1 = ab_grid[0, combo[0] - 1] # from ab_grid, coop 0 and value 0 of combo N
        val2 = ab_grid[1, combo[1] - 1] # from ab_grid, coop 1 and value 1 of combo N
        val3 = ab_grid[2, combo[2] - 1] # from ab_grid, coop 2 and value 2 of combo N
        val4 = ab_grid[3, combo[3] - 1] # from ab_grid, coop 3 and value 3 of combo N

        prob = val1 * val2 * val3 * val4 # multipy them together to get the probability of that ability
        
        prob_combo_list.append([list(combo), [prob]]) # append the list of lists
 
    largest_probability = max(prob_combo_list, key=lambda x: x[1][0]) # find the largest probabilty
    
    best_combo = largest_probability[0] # the best ability combination
    # find accuracy
    true_profile = ent_set.chickens[competitor_ID].MyAbility
    accuracy = jnp.mean(true_profile == jnp.array(best_combo)) * 100

    return best_combo, largest_probability, accuracy

def AbilityBelief_Print(chick_ID, M, N, ent_set): # print the ability profile
   
   # Print Title Block
    print("=" * 80)
    print(f"P(chicken {chick_ID} has the printed ability profile | obs history)]")
    print("Each row = one chicken j, columns = coops")
    print("=" * 80)

    # Print Header
    header = f"{'j':<3}"
    for coop in range(N):
        header += f"|Ab{coop:<1}"
    header += f"|Prob of Ab Score"
    header += f"|Actual Ability"
    header += f"|Accuracy"
    print(header)

    # Rows
    for j in range(M):
        row_str = f"{j:<5}" 
        best_combo, largest_probability, accuracy = P_ability_profile(chick_ID, j, ent_set, N) # run the P_Abilty_profile function
        prob_val = largest_probability[1][0]
        for coop in range(N):
            score = best_combo[coop] 
            row_str += f"{score:<4}"
        
        row_str += f"{prob_val:<15.4f}"
        j_abilities = [int(x) for x in ent_set.chickens[j].MyAbility] # print true ability
        row_str += f"j={str(j_abilities):<15}" 
        row_str += f"{accuracy}%"
        
        # Now print the completed row
        print(row_str)

# =====================
# Main Loop
# =====================
def main():
    T = 10 # 1000 tournaments    
    M = 64 # chickens
    N = 4 # regions/ coops/ skills
    k = 4

    env, ent_set = Events.init(M, N, T) # spawn the environment and entity set
    king_list = []
    for T_i in range(T):
        env, ent_set = Events.new_tournament(env, ent_set, M, N)    
        env, ent_set = Events.tournament(env, ent_set, T_i, M, N, k)
        env, ent_set, current_kings = Events.close_tournament(env, ent_set, T_i, M, N, k) 
        king_list.append(current_kings)
        ent_set, emp_king_list = CountCrowns(king_list, M, ent_set)      
    
    #print(king_list)
    ent_set, emp_king_list = CountCrowns(king_list, M, ent_set)

    print('[(chick) (# of crowns)]')
    print(emp_king_list)
    
    
    ### p(i is king| Observation History)
    P_i_king_Print(ent_set, N, M)
    print("Press any key to continue...")
    msvcrt.getch()

    for i in range(M):    
        PivsjCleanPrint(i, ent_set, N, M) # show P(ibeatsj|ObservationHistory)        
    
    ### p(AbilityProfileChickenJ|ObservationHistory)
        AbilityBelief_Print(i, M, N, ent_set)
        print("Press any key to continue...")
        msvcrt.getch()

if __name__ == "__main__":
    main()