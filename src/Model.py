from mesa import Agent
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class SchellingAgent(Agent):
    # 1 Initialization
    def __init__(self, pos, model, agent_type):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type

    # 2 Step function
    def step(self):
        similar = 0
        # 3 Calculate the number of similar neighbours
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            if neighbor.type == self.type:
                similar += 1

        # 4 Move to a random empty location if unhappy
        if similar < self.model.homophily:
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1


class Schelling(Model):
    def __init__(self, height, width, density, minority_pc, homophily):

        self.height = height  # vertical axis of the grid
        self.width = width  # horizantal axis of the grid
        self.density = density  # pop density of the agent in the system 0<=val<=1
        self.minority_pc = minority_pc  # the ratio of blue to red 0<=val<=1
        self.homophily = homophily  # represents agent happiness 0<=val<=8

        self.grid = SingleGrid(height, width, torus=True)
        self.schedule = RandomActivation(self)

        self.datacollector = DataCollector(
            {"happy": "happy"},  # Model-level count of happy agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]},
        )

        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            # probability of having a blank cell inversely proportional to pop density
            if self.random.random() < self.density:
                if self.random.random() < self.minority_pc:
                    # minority agent type
                    agent_type = 1
                else:
                    # majority agent type
                    agent_type = 0

                agent = SchellingAgent((x, y), self, agent_type)
                self.grid.position_agent(agent, (x, y))
                self.schedule.add(agent)

        # turn the model on and off
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """
        self.happy = 0  # Reset counter of happy agents
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        # check if all the agents are happy; if they are stop the model from running
        if self.happy == self.schedule.get_agent_count():
            self.running = False
