"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from datetime import datetime, timedelta

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.temporal_operations import (
    extract_date_strings_from_edge,
    prepare_edges_for_invalidation,
    prepare_invalidation_context,
)


# Helper function to create test data
def create_test_data():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now, group_id='1')
    node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now, group_id='1')
    node3 = EntityNode(uuid='3', name='Node3', labels=['Person'], created_at=now, group_id='1')

    # Create edges
    existing_edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='KNOWS',
        fact='Node1 knows Node2',
        created_at=now,
        group_id='1',
    )
    existing_edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='2',
        target_node_uuid='3',
        name='LIKES',
        fact='Node2 likes Node3',
        created_at=now,
        group_id='1',
    )
    new_edge1 = EntityEdge(
        uuid='e3',
        source_node_uuid='1',
        target_node_uuid='3',
        name='WORKS_WITH',
        fact='Node1 works with Node3',
        created_at=now,
        group_id='1',
    )
    new_edge2 = EntityEdge(
        uuid='e4',
        source_node_uuid='1',
        target_node_uuid='2',
        name='DISLIKES',
        fact='Node1 dislikes Node2',
        created_at=now,
        group_id='1',
    )

    return {
        'nodes': [node1, node2, node3],
        'existing_edges': [existing_edge1, existing_edge2],
        'new_edges': [new_edge1, new_edge2],
    }


def test_prepare_edges_for_invalidation_basic():
    test_data = create_test_data()

    existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
        test_data['existing_edges'], test_data['new_edges'], test_data['nodes']
    )

    assert len(existing_edges_pending_invalidation) == 2
    assert len(new_edges_with_nodes) == 2

    # Check if the edges are correctly associated with nodes
    for edge_with_nodes in existing_edges_pending_invalidation + new_edges_with_nodes:
        assert isinstance(edge_with_nodes[0], EntityNode)
        assert isinstance(edge_with_nodes[1], EntityEdge)
        assert isinstance(edge_with_nodes[2], EntityNode)


def test_prepare_edges_for_invalidation_no_existing_edges():
    test_data = create_test_data()

    existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
        [], test_data['new_edges'], test_data['nodes']
    )

    assert len(existing_edges_pending_invalidation) == 0
    assert len(new_edges_with_nodes) == 2


def test_prepare_edges_for_invalidation_no_new_edges():
    test_data = create_test_data()

    existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
        test_data['existing_edges'], [], test_data['nodes']
    )

    assert len(existing_edges_pending_invalidation) == 2
    assert len(new_edges_with_nodes) == 0


def test_prepare_edges_for_invalidation_missing_nodes():
    test_data = create_test_data()

    # Remove one node to simulate a missing node scenario
    nodes = test_data['nodes'][:-1]

    existing_edges_pending_invalidation, new_edges_with_nodes = prepare_edges_for_invalidation(
        test_data['existing_edges'], test_data['new_edges'], nodes
    )

    assert len(existing_edges_pending_invalidation) == 1
    assert len(new_edges_with_nodes) == 1


def test_prepare_invalidation_context():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now, group_id='1')
    node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now, group_id='1')
    node3 = EntityNode(uuid='3', name='Node3', labels=['Person'], created_at=now, group_id='1')

    # Create edges
    edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='KNOWS',
        fact='Node1 knows Node2',
        created_at=now,
        group_id='1',
    )
    edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='2',
        target_node_uuid='3',
        name='LIKES',
        fact='Node2 likes Node3',
        created_at=now,
        group_id='1',
    )

    # Create NodeEdgeNodeTriplet objects
    existing_edge = (node1, edge1, node2)
    new_edge = (node2, edge2, node3)

    # Prepare test input
    existing_edges = [existing_edge]
    new_edges = [new_edge]

    # Create a current episode and previous episodes
    current_episode = EpisodicNode(
        name='Current Episode',
        content='This is the current episode content.',
        created_at=now,
        valid_at=now,
        source=EpisodeType.message,
        source_description='Test episode for unit testing',
        group_id='1',
    )
    previous_episodes = [
        EpisodicNode(
            name='Previous Episode 1',
            content='This is the content of previous episode 1.',
            created_at=now - timedelta(days=1),
            valid_at=now - timedelta(days=1),
            source=EpisodeType.message,
            source_description='Test previous episode 1 for unit testing',
            group_id='1',
        ),
        EpisodicNode(
            name='Previous Episode 2',
            content='This is the content of previous episode 2.',
            created_at=now - timedelta(days=2),
            valid_at=now - timedelta(days=2),
            source=EpisodeType.message,
            source_description='Test previous episode 2 for unit testing',
            group_id='1',
        ),
    ]

    # Call the function
    result = prepare_invalidation_context(
        existing_edges, new_edges, current_episode, previous_episodes
    )

    # Assert the result
    assert isinstance(result, dict)
    assert 'existing_edges' in result
    assert 'new_edges' in result
    assert 'current_episode' in result
    assert 'previous_episodes' in result
    assert len(result['existing_edges']) == 1
    assert len(result['new_edges']) == 1
    assert result['current_episode'] == current_episode.content
    assert len(result['previous_episodes']) == 2

    # Check the format of the existing edge
    existing_edge_str = result['existing_edges'][0]
    assert edge1.uuid in existing_edge_str
    assert node1.name in existing_edge_str
    assert edge1.name in existing_edge_str
    assert node2.name in existing_edge_str
    assert edge1.fact in existing_edge_str

    # Check the format of the new edge
    new_edge_str = result['new_edges'][0]
    assert edge2.uuid in new_edge_str
    assert node2.name in new_edge_str
    assert edge2.name in new_edge_str
    assert node3.name in new_edge_str
    assert edge2.fact in new_edge_str


def test_prepare_invalidation_context_empty_input():
    now = datetime.now()
    current_episode = EpisodicNode(
        name='Current Episode',
        content='Empty episode',
        created_at=now,
        valid_at=now,
        source=EpisodeType.message,
        source_description='Test empty episode for unit testing',
        group_id='1',
    )
    result = prepare_invalidation_context([], [], current_episode, [])
    assert isinstance(result, dict)
    assert 'existing_edges' in result
    assert 'new_edges' in result
    assert 'current_episode' in result
    assert 'previous_episodes' in result
    assert len(result['existing_edges']) == 0
    assert len(result['new_edges']) == 0
    assert result['current_episode'] == current_episode.content
    assert len(result['previous_episodes']) == 0


def test_prepare_invalidation_context_sorting():
    now = datetime.now()

    # Create nodes
    node1 = EntityNode(uuid='1', name='Node1', labels=['Person'], created_at=now, group_id='1')
    node2 = EntityNode(uuid='2', name='Node2', labels=['Person'], created_at=now, group_id='1')

    # Create edges with different timestamps
    edge1 = EntityEdge(
        uuid='e1',
        source_node_uuid='1',
        target_node_uuid='2',
        name='KNOWS',
        fact='Node1 knows Node2',
        created_at=now,
        group_id='1',
    )
    edge2 = EntityEdge(
        uuid='e2',
        source_node_uuid='2',
        target_node_uuid='1',
        name='LIKES',
        fact='Node2 likes Node1',
        created_at=now + timedelta(hours=1),
        group_id='1',
    )

    edge_with_nodes1 = (node1, edge1, node2)
    edge_with_nodes2 = (node2, edge2, node1)

    # Prepare test input
    existing_edges = [edge_with_nodes1, edge_with_nodes2]

    # Create a current episode and previous episodes
    current_episode = EpisodicNode(
        name='Current Episode',
        content='This is the current episode content.',
        created_at=now,
        valid_at=now,
        source=EpisodeType.message,
        source_description='Test episode for unit testing',
        group_id='1',
    )
    previous_episodes = [
        EpisodicNode(
            name='Previous Episode',
            content='This is the content of a previous episode.',
            created_at=now - timedelta(days=1),
            valid_at=now - timedelta(days=1),
            source=EpisodeType.message,
            source_description='Test previous episode for unit testing',
            group_id='1',
        ),
    ]

    # Call the function
    result = prepare_invalidation_context(existing_edges, [], current_episode, previous_episodes)

    # Assert the result
    assert len(result['existing_edges']) == 2
    assert edge2.uuid in result['existing_edges'][0]  # The newer edge should be first
    assert edge1.uuid in result['existing_edges'][1]  # The older edge should be second
    assert result['current_episode'] == current_episode.content
    assert len(result['previous_episodes']) == 1
    assert result['previous_episodes'][0] == previous_episodes[0].content


class TestExtractDateStringsFromEdge(unittest.TestCase):
    def generate_entity_edge(self, valid_at, invalid_at):
        return EntityEdge(
            source_node_uuid='1',
            target_node_uuid='2',
            name='KNOWS',
            fact='Node1 knows Node2',
            created_at=datetime.now(),
            valid_at=valid_at,
            invalid_at=invalid_at,
            group_id='1',
        )

    def test_both_dates_present(self):
        edge = self.generate_entity_edge(datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 2, 12, 0))
        result = extract_date_strings_from_edge(edge)
        expected = 'Start Date: 2024-01-01T12:00:00 (End Date: 2024-01-02T12:00:00)'
        self.assertEqual(result, expected)

    def test_only_valid_at_present(self):
        edge = self.generate_entity_edge(datetime(2024, 1, 1, 12, 0), None)
        result = extract_date_strings_from_edge(edge)
        expected = 'Start Date: 2024-01-01T12:00:00'
        self.assertEqual(result, expected)

    def test_only_invalid_at_present(self):
        edge = self.generate_entity_edge(None, datetime(2024, 1, 2, 12, 0))
        result = extract_date_strings_from_edge(edge)
        expected = ' (End Date: 2024-01-02T12:00:00)'
        self.assertEqual(result, expected)

    def test_no_dates_present(self):
        edge = self.generate_entity_edge(None, None)
        result = extract_date_strings_from_edge(edge)
        expected = ''
        self.assertEqual(result, expected)


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])
